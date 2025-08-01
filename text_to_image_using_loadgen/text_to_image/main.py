"""
mlperf inference benchmarking tool
"""


from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import warnings
warnings.simplefilter("ignore", category=FutureWarning)

from tqdm import auto as tqdm_lib
import argparse
import array
import logging
import os
import sys
import torch.multiprocessing as mp
import threading
import time

import mlperf_loadgen as lg
import numpy as np
import torch

import dataset
import coco
post_proc = None


logging.basicConfig(level=logging.INFO)
log = logging.getLogger("main")

NANO_SEC = 1e9
MILLI_SEC = 1000

SUPPORTED_DATASETS = {
    "coco-1024": (
        coco.Coco_stageai, #TODO: Coco_stageai or appropriate class
        dataset.preprocess,
        coco.PostProcessCoco(),
        {"image_size": [3, 1024, 1024]},
    )
}


SUPPORTED_PROFILES = {
    "defaults": {
        "dataset": "coco-1024",
        "backend": "pytorch",
        "model-name": "stable-diffusion-xl",
    },
    "stable-diffusion-xl-stageai": {
        "dataset": "coco-1024",
        "backend": "stageai",
        "model-name": "stable-diffusion-xl",
    },
}

SCENARIO_MAP = {
    "Server":  lg.TestScenario.Server,
    "SingleStream": lg.TestScenario.SingleStream,
    "Offline": lg.TestScenario.Offline,
}


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cache-dir', type=str, default=None, help='Path to the cache directory')
    parser.add_argument("--dataset", choices=SUPPORTED_DATASETS.keys(), help="dataset")
    parser.add_argument("--dataset-path", required=True, help="path to the dataset")
    parser.add_argument(
        "--profile", choices=SUPPORTED_PROFILES.keys(), help="standard profiles"
    )
    parser.add_argument(
        "--scenario",
        default="SingleStream",
        help="mlperf benchmark scenario, one of " + str(list(SCENARIO_MAP.keys())),
    )
    parser.add_argument(
        "--max-batchsize",
        type=int,
        default=1,
        help="max batch size in a single inference",
    )
    parser.add_argument("--threads", default=1, type=int, help="threads")
    parser.add_argument("--accuracy", action="store_true", help="enable accuracy pass")
    parser.add_argument(
        "--find-peak-performance",
        action="store_true",
        help="enable finding peak performance pass",
    )
    parser.add_argument("--backend", help="Name of the backend")
    parser.add_argument("--model-name", help="Name of the model")
    parser.add_argument("--output", default="output", help="test results")
    parser.add_argument("--qps", type=float, help="target qps")
    parser.add_argument("--model-path", help="Path to model weights")

    parser.add_argument("--compiled_text_encoder_dir", help="Path to text_encoder")
    parser.add_argument("--compiled_text_encoder_2_dir", help="Path to text_encoder_2")
    parser.add_argument("--compiled_unet_dir", help="Path to unet")
    parser.add_argument("--compiled_vae_decoder_dir", help="Path to vae_decoder")
    parser.add_argument("--latent_dir", default=f"{os.path.dirname(os.path.abspath(__file__))}/tools/latents.pt", help="Path to vae_decoder")

    parser.add_argument("--model_path", help="Path to the sdxl model pipe")
    parser.add_argument("--axs_device_id", help="Customised Device IDs from AXS")

    parser.add_argument(
        "--dtype",
        default="fp32",
        choices=["fp32", "fp16", "bf16"],
        help="dtype of the model",
    )
    parser.add_argument(
        "--device",
        default="cuda",
        choices=["cuda", "cpu"],
        help="device to run the benchmark",
    )
    parser.add_argument(
        "--latent-framework",
        default="torch",
        choices=["torch", "numpy"],
        help="framework to load the latents",
    )

    # file to use mlperf rules compliant parameters
    parser.add_argument(
        "--mlperf_conf", default="mlperf.conf", help="mlperf rules config"
    )
    # file for user LoadGen settings such as target QPS
    parser.add_argument(
        "--user_conf",
        default="user.conf",
        help="user config for user LoadGen settings such as target QPS",
    )
    # file for LoadGen audit settings
    parser.add_argument(
        "--audit_conf", default="audit.config", help="config for LoadGen audit settings"
    )
    # arguments to save images
    # pass this argument for official submission
    # parser.add_argument("--output-images", action="store_true", help="Store a subset of the generated images")
    # do not modify this argument for official submission
    parser.add_argument("--ids-path", help="Path to caption ids", default="tools/sample_ids.txt")

    # below will override mlperf rules compliant settings - don't use for official submission
    parser.add_argument("--time", type=int, help="time to scan in seconds")
    parser.add_argument("--count", type=int, help="dataset items to use")
    parser.add_argument("--debug", action="store_true", help="debug")
    parser.add_argument("--use-dual", action="store_true", help="pair/dual")
    parser.add_argument("--num-devices", type=int, default=1, help="provide num devices")

    parser.add_argument(
        "--performance-sample-count", type=int, help="performance sample count", default=5000
    )
    parser.add_argument(
        "--max-latency", type=float, help="mlperf max latency in pct tile"
    )
    parser.add_argument(
        "--samples-per-query",
        default=8,
        type=int,
        help="mlperf multi-stream samples per query",
    )

    # Server mode specific arguments
    parser.add_argument(
        "--batch-timeout-threshold",
        type=float,
        default=5,
        help="Timeout threshold for server mode batching (in seconds)",
    )

    args = parser.parse_args()

    # don't use defaults in argparser. Instead we default to a dict, override that with a profile
    # and take this as default unless command line give
    defaults = SUPPORTED_PROFILES["defaults"]

    if args.profile:
        profile = SUPPORTED_PROFILES[args.profile]
        defaults.update(profile)
    for k, v in defaults.items():
        kc = k.replace("-", "_")
        if getattr(args, kc) is None:
            setattr(args, kc, v)

    if args.scenario not in SCENARIO_MAP:
        parser.error("valid scanarios:" + str(list(SCENARIO_MAP.keys())))
    return args


def get_backend(backend, **kwargs):
    if backend == "stageai":
        from backend_stageai import BackendStageai
        backend = BackendStageai(**kwargs)
    else:
        raise ValueError("unknown backend: " + backend)
    return backend

class Item:
    """An item that we queue for processing by the thread pool."""

    def __init__(self, query_id, content_id, img=None):
        self.query_id = query_id
        self.content_id = content_id
        self.img = img
        # self.inputs = inputs
        self.start = time.time()

def worker_process_func(node_id, tasks_queue, response_queue, worker_config, init_queue, loaded_0):  # Add init_queue arg
    """Worker process function that initializes backend and processes tasks."""
    # Set up logging for the worker process
    logging.basicConfig(level=logging.INFO)
    log = logging.getLogger(f"worker_{node_id}")

    # Reconstruct dataset and post-processor in the worker process
    dataset_class = worker_config['dataset_class']
    ds = dataset_class(data_path=worker_config['dataset_path'])


    status = "done"  # Default to success
    try:
        if worker_config['use_dual_device']:
            device_id = worker_config['device_ids'][node_id*2]
            device_id2 = worker_config['device_ids'][node_id*2 + 1]
        else:
            device_id = worker_config['device_ids'][node_id]
            device_id2 = None

        # Set CUDA device BEFORE importing the backend
        torch.cuda.set_device(device_id)

        from backend_stageai import BackendStageai as BackendClass
        backend = BackendClass(
            batch_size=worker_config['batch_size'],
            device=f'cuda:{device_id}',
            latent_dir=worker_config["latent_dir"],
            cache_dir=worker_config['cache_dir']
        )

        backend.load()
        log.info(f"Backend initialized on cuda:{device_id}")
        if node_id == 0:
            loaded_0.value = 1
        # Warm-up prediction

        log.info("Warming up backend")
        backend.warmup()
        log.info("Warming up complete")

        #backend.predict(["krazy people working on MLPerf Submission"], output_type="latent")
        #for bs in range(worker_config['batch_size'], 0, -1):
        #   backend.predict(["krazy people working on MLPerf Submission"]*bs)
        # for _ in range(3):
        #     backend.predict(["krazy people working on MLPerf Submission"]*worker_config['batch_size'])
        #backend.predict(["krazy people working on MLPerf Submission"]*worker_config['batch_size'])


    except Exception as e:
        if node_id == 0:
            loaded_0.value = 2
        log.error(f"Failed to initialize backend: {e}")
        status = "error"

    finally:
        init_queue.put(status)  # Always signal status, even on error

    if status == "error":
        return  # Exit worker early on init failure

    # Process tasks (rest of the function unchanged)
    while True:
        try:
            task = tasks_queue.get(timeout=1)
            if task is None:  # Shutdown signal
                break

            qitem = task

            try:
                captions = ds.get_captions(qitem.content_id)
                results = backend.predict(captions)


                # Send results back to main process
                for query_id, res, content_id in zip(qitem.query_id, results, qitem.content_id):
                    response_queue.put((query_id, res, content_id))


            except Exception as ex:
                log.error("Failed on content_id=%s, %s", qitem.content_id, ex)
                # Send empty results for failed queries
                for query_id, content_id in zip(qitem.query_id, qitem.content_id):
                    response_queue.put((query_id, torch.empty(0,0,0), content_id))

        except Exception as e:
            if str(e) != "":  # Ignore timeout exceptions
                log.error(f"Worker {node_id} error: {e}")


    while True:
        if response_queue.empty():
            time.sleep(5)
            break
        else:
            time.sleep(5)

    log.info(f"Worker {node_id} shutting down")


class RunnerBase:
    def __init__(self, ds, post_proc=None, max_batchsize=128):
        self._ds = ds
        self.post_process = post_proc
        self.take_accuracy = False
        self.max_batchsize = max_batchsize
        self.result_timing = []

    def start_run(self, result_dict, take_accuracy):
        self.result_dict = result_dict
        self.take_accuracy = take_accuracy
        if hasattr(self.post_process, 'start'):
            self.post_process.start()

    def get_ds(self):
        return self._ds

    def run_one_item(self, backend, qitem):
        # Not used in multiprocessing version
        pass

    def enqueue(self, query_samples):
        idx = [q.index for q in query_samples]
        query_id = [q.id for q in query_samples]
        print(len(query_samples), self.max_batchsize)
        if len(query_samples) < self.max_batchsize:
            self.run_one_item(Item(query_id, idx))
        else:
            bs = self.max_batchsize
            for i in range(0, len(idx), bs):
                self.run_one_item(Item(query_id[i:i+bs], idx[i:i+bs]))

    def finish(self):
        pass

class QueueRunner(RunnerBase):
    def __init__(self, ds, config):
        super().__init__(ds, config['post_proc'], config['max_batchsize'])

        # Prepare worker configuration with only picklable data
        self.worker_config = {
            'batch_size': config['max_batchsize'],
            'use_dual_device': config['use_dual_device'],
            'device_ids': config['device_ids'],
            'dataset_path': config['dataset_path'],  # Already absolute
            'dataset_class': SUPPORTED_DATASETS[config['dataset_name']][0],
            'compiled_text_encoder_dir': config.get('compiled_text_encoder_dir'),
            'compiled_text_encoder_2_dir': config.get('compiled_text_encoder_2_dir'),
            'compiled_unet_dir': config.get('compiled_unet_dir'),
            'compiled_vae_decoder_dir': config.get('compiled_vae_decoder_dir'),
            'latent_dir': config.get('latent_dir'),
            'cache_dir': config.get('cache_dir'),
        }

        self.progress =  tqdm_lib.tqdm(total=0)

        self.tasks_queue = mp.Queue()
        self.response_queue = mp.Queue()
        self.init_queue = mp.Queue()  # New: Queue for init synchronization
        self.workers = []
        self.response_thread = None
        self.start_workers()
        self.start_response_handler()

    def start_workers(self):
        # Calculate number of processes based on device configuration
        num_devices = len(self.worker_config['device_ids'])
        num_processes = num_devices // 2 if self.worker_config['use_dual_device'] else num_devices

        loaded_0 = mp.Value('i', 0)

        for node_id in range(num_processes):
            worker = mp.Process(
                target=worker_process_func,
                args=(node_id, self.tasks_queue, self.response_queue, self.worker_config, self.init_queue, loaded_0)  # Pass init_queue
            )
            worker.start()
            self.workers.append(worker)
            time.sleep(1)  # Small delay between process starts (optional, but retained)
            if node_id == 0:
                while loaded_0.value == 0:
                   time.sleep(1)  # Small delay between process starts (optional, but retained)
                if loaded_0.value == 2:
                   self.finish()
                   raise Exception("Backend loading failed. Check logs for details.")
            else:
                time.sleep(1)
                

        # New: Wait for all workers to signal init complete
        for _ in range(num_processes):
            status = self.init_queue.get()
            if status == "error":
                log.error("A worker failed initialization. Check logs for details.")
                self.finish()  # Clean up workers
                sys.exit(1)  # Or raise an exception
            elif status == "done":
                log.info("Worker signaled ready.")
            else:
                log.warning(f"Unexpected status from worker: {status}")

    def start_response_handler(self):
        """Start a thread to handle responses from worker processes."""
        self.response_thread = threading.Thread(target=self.handle_responses)
        self.response_thread.daemon = True
        self.response_thread.start()

    def handle_responses(self):
        """Handle responses from worker processes and complete queries."""
        while True:
            try:
                response = self.response_queue.get(timeout=1)
                if response is None:  # Shutdown signal
                    break

                query_id, tensor, content_id = response
                image_array = post_proc([tensor], [content_id])[0]
                del tensor
                if image_array.size > 0:
                    image_bytes = image_array.tobytes()
                    response_array = array.array("B", image_bytes)
                    bi = response_array.buffer_info()
                    lg.QuerySamplesComplete([lg.QuerySampleResponse(query_id, bi[0], bi[1])])
                else:
                    # Handle failed queries
                    empty_array = array.array("B", [])
                    bi = empty_array.buffer_info()
                    lg.QuerySamplesComplete([lg.QuerySampleResponse(query_id, bi[0], bi[1])])

                self.progress.update(1)
                #self.progress.update(len(response))

            except Exception as e:
                if str(e) != "":  # Ignore timeout exceptions
                    log.error(f"Response handler error: {e}")

        self.progress.close()


    def enqueue(self, query_samples):
        batch_size = self.max_batchsize

        #print(f"query_samples: {len(query_samples)}")
        self.progress.total += len(query_samples)
        self.progress.refresh()
        for i in range(0, len(query_samples), batch_size):
            batch = query_samples[i:i+batch_size]
            query_ids = [sample.id for sample in batch]
            indices = [sample.index for sample in batch]
            item = Item(query_ids, indices)
            self.tasks_queue.put(item)

    def finish(self):
        # Send shutdown signal to all workers
        for _ in self.workers:
            self.tasks_queue.put(None)

        # Wait for all workers to finish
        for worker in self.workers:
            worker.join(timeout=10)
            if worker.is_alive():
                worker.terminate()
                worker.join()

        # Stop response handler
        self.response_queue.put(None)
        if self.response_thread:
            self.response_thread.join(timeout=5)


class QueueServerRunner(QueueRunner):
    def __init__(self, ds, config):
        super().__init__(ds, config)
        # Server-specific configuration
        self.batch_timeout_threshold = config.get('batch_timeout_threshold', 0.01)  # 10ms default
        self.enable_batcher = self.max_batchsize > 1 and self.batch_timeout_threshold > 0

        if self.enable_batcher:
            self.batcher_queue = mp.Queue()
            self.batcher_thread = threading.Thread(target=self.batch_samples_loop, daemon=True)
            self.batcher_thread.start()
            log.info(f"Server batching enabled with max_batchsize={self.max_batchsize}, timeout={self.batch_timeout_threshold}s")
        else:
            log.info(f"Server batching disabled (bs={self.max_batchsize})")

    def batch_samples_loop(self):
        """
        Batching loop that accumulates queries and forms batches based on:
        1. Maximum batch size reached
        2. Timeout threshold exceeded
        """
        batched_samples = []
        timeout_stamp = time.time()

        while True:
            # Check if we should send a batch
            if len(batched_samples) > 0 and (
                len(batched_samples) >= self.max_batchsize or
                time.time() - timeout_stamp >= self.batch_timeout_threshold
            ):
                # Form and send batch
                batch_to_send = batched_samples[:self.max_batchsize]
                query_ids = [sample.id for sample in batch_to_send]
                indices = [sample.index for sample in batch_to_send]
                log.info(f"Formed batch of {len(batch_to_send)} samples")

                # Create batch item and put in tasks queue
                item = Item(query_ids, indices)
                self.tasks_queue.put(item)

                # Keep remaining samples for next batch
                batched_samples = batched_samples[self.max_batchsize:]
                timeout_stamp = time.time()

            try:
                # Try to get new samples with timeout
                sample = self.batcher_queue.get(timeout=self.batch_timeout_threshold)
                if sample is None:  # Shutdown signal
                    # Process any remaining samples
                    if batched_samples:
                        query_ids = [s.id for s in batched_samples]
                        indices = [s.index for s in batched_samples]
                        item = Item(query_ids, indices)
                        self.tasks_queue.put(item)
                    break

                batched_samples.append(sample)

                # If this is the first sample in a new batch, reset timeout
                if len(batched_samples) == 1:
                    timeout_stamp = time.time()

            except:  # Queue.Empty
                # Timeout occurred, loop will check if batch should be sent
                continue

    def enqueue(self, query_samples):
        """
        Override enqueue to support batching for server mode
        """

        if self.enable_batcher:
            # Send samples to batcher queue one by one
            self.progress.total += len(query_samples)
            self.progress.refresh()
            for sample in query_samples:
                self.batcher_queue.put(sample)
        else:
            # For bs=1 or batching disabled, use parent implementation
            super().enqueue(query_samples)

    def finish(self):
        """
        Override finish to properly shutdown batcher thread
        """
        if self.enable_batcher:
            # Signal batcher thread to stop
            self.batcher_queue.put(None)
            self.batcher_thread.join()

        # Call parent finish
        super().finish()


import os
import shutil
import sys
from functools import wraps

REQUIRED_SPACE = 30 * 1024 * 1024 * 1024  # 30GB in bytes

def check_disk_space(path, required_space):
    """Checks if there's enough free space at the given path."""
    total, used, free = shutil.disk_usage(path)
    return free >= required_space

def check_space(func):
    """Decorator to check disk space before running a function. Exits if insufficient space."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        current_path = os.getcwd()  # Get the current working directory
        if check_disk_space(current_path, REQUIRED_SPACE):
            return func(*args, **kwargs)
        else:
            print("Insufficient disk space! Please free up space before running.")
            sys.exit(1)  # Exit the program with an error code
    return wrapper

@check_space
def run_space_checker():
    # Placeholder for your accuracy calculation code
    print("Checking accuracy process")


def main():
    # Set multiprocessing start method before any multiprocessing objects are created
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        # Already set, ignore
        pass

    args = get_args()

    log.info(args)

    if args.backend == "stageai":
        if args.scenario == "Offline":
            if args.accuracy:
                args.max_batchsize = 10
            else:
                args.max_batchsize = 16
        elif  args.scenario == "Server":
            args.max_batchsize = 16
        elif args.scenario == "SingleStream":
            args.max_batchsize = 1

    if args.dtype == "fp16":
        dtype = torch.float16
    elif args.dtype == "bf16":
        dtype = torch.bfloat16
    else:
        dtype = torch.float32

    # --count applies to accuracy mode only and can be used to limit the number of images
    # for testing.
    count_override = False
    count = args.count
    if count:
        count_override = True

    # Convert all file paths to absolute paths before changing directory
    mlperf_conf = os.path.abspath(args.mlperf_conf)
    if not os.path.exists(mlperf_conf):
        log.error("{} not found".format(mlperf_conf))
        sys.exit(1)

    user_conf = os.path.abspath(args.user_conf)
    if not os.path.exists(user_conf):
        log.error("{} not found".format(user_conf))
        sys.exit(1)

    audit_config = os.path.abspath(args.audit_conf)

    # dataset to use
    global post_proc
    dataset_class, pre_proc, post_proc, kwargs = SUPPORTED_DATASETS[args.dataset]
    # Convert dataset path to absolute path before any directory changes
    dataset_path_abs = os.path.abspath(args.dataset_path)
    ds = dataset_class(
        data_path=dataset_path_abs,
    )

    final_results = {
        "runtime": "StageAI-SUT",
        "version": "unknown",
        "time": int(time.time()),
        "args": vars(args),
        "cmdline": str(args),
    }

    if args.accuracy:
        run_space_checker()
        ids_path = os.path.abspath(args.ids_path)
        with open(ids_path) as f:
            saved_images_ids = [int(_) for _ in f.readlines()]

    if args.output:
        output_dir = os.path.abspath(args.output)
        os.makedirs(output_dir, exist_ok=True)
        os.chdir(output_dir)


    scenario = SCENARIO_MAP[args.scenario]

    RUNNER_MAP = {
        'stageai': {
            lg.TestScenario.Server:  QueueServerRunner,  # Use server runner for server scenario
            lg.TestScenario.SingleStream: QueueRunner,
            lg.TestScenario.Offline: QueueRunner,
        }
    }
    runner_map = RUNNER_MAP[args.backend]

    runner_class = runner_map[scenario]
    if scenario == "SingleStream":
        args.num_devices = 1
        if args.use_dual:
            args.num_devices = 2

    # config for potential future use of different artifacts
    # Parse device IDs
    try:
        import re
        device_ids = [int(id) for id in re.split(r'[,+]', args.axs_device_id)]
    except (ValueError, AttributeError, TypeError):
        device_ids = list(range(args.num_devices))


    # Handle latent_dir - it might already be absolute (default value)
    latent_dir_abs = None
    if args.latent_dir:
        if os.path.isabs(args.latent_dir):
            latent_dir_abs = args.latent_dir
        else:
            latent_dir_abs = os.path.abspath(args.latent_dir)

    config = {
        "compiled_text_encoder_dir": os.path.abspath(args.compiled_text_encoder_dir) if args.compiled_text_encoder_dir else None,
        "compiled_text_encoder_2_dir": os.path.abspath(args.compiled_text_encoder_2_dir) if args.compiled_text_encoder_2_dir else None,
        "compiled_unet_dir": os.path.abspath(args.compiled_unet_dir) if args.compiled_unet_dir else None,
        "compiled_vae_decoder_dir": os.path.abspath(args.compiled_vae_decoder_dir) if args.compiled_vae_decoder_dir else None,
        "latent_dir": latent_dir_abs,
        "post_proc": post_proc,
        "max_batchsize": args.max_batchsize,
        "use_dual_device": args.use_dual,
        "device_ids": device_ids,
        "batch_size": args.max_batchsize,
        "num_nodes": args.num_devices,
        "model_path": args.model_path,  # This is typically a HuggingFace model ID, not a path
        "backend_type": args.backend,
        "dataset_name": args.dataset,
        "dataset_path": dataset_path_abs,  # Use absolute path
        "batch_timeout_threshold": args.batch_timeout_threshold,  # Add server mode batching timeout
        "cache_dir": args.cache_dir,
    }

    runner = runner_class(
        ds, config
    )

    def issue_queries(query_samples):
        runner.enqueue(query_samples)

    def flush_queries():
        pass

    log_output_settings = lg.LogOutputSettings()
    log_output_settings.outdir = output_dir
    log_output_settings.copy_summary_to_stdout = False
    log_settings = lg.LogSettings()
    log_settings.enable_trace = args.debug
    log_settings.log_output = log_output_settings

    settings = lg.TestSettings()
    settings.FromConfig(mlperf_conf, args.model_name, args.scenario)
    #settings.FromConfig(user_conf, args.model_name, args.scenario)
    settings.scenario = scenario
    settings.mode = lg.TestMode.PerformanceOnly
    if args.accuracy:
        settings.mode = lg.TestMode.AccuracyOnly
    if args.find_peak_performance:
        settings.mode = lg.TestMode.FindPeakPerformance

    if args.time:
        # override the time we want to run
        settings.min_duration_ms = args.time * MILLI_SEC
        settings.max_duration_ms = args.time * MILLI_SEC

    if args.qps:
        qps = float(args.qps)
        settings.server_target_qps = qps
        settings.offline_expected_qps = qps

    if count_override:
        settings.min_query_count = count
        #settings.max_query_count = count

    if args.samples_per_query:
        settings.multi_stream_samples_per_query = args.samples_per_query
    if args.max_latency:
        settings.server_target_latency_ns = int(args.max_latency * NANO_SEC)
        settings.multi_stream_expected_latency_ns = int(args.max_latency * NANO_SEC)

    #
    # make one pass over the dataset to validate accuracy
    #
    count = ds.get_item_count()
    performance_sample_count = (
        args.performance_sample_count
        if args.performance_sample_count
        else min(count, 500)
    )
    print(performance_sample_count)
    sut = lg.ConstructSUT(issue_queries, flush_queries)
    qsl = lg.ConstructQSL(
        count, performance_sample_count, ds.load_query_samples, ds.unload_query_samples
    )

    log.info("starting {}".format(scenario))
    result_dict = {"scenario": str(scenario)}
    log.info(f"accuracy {args.accuracy}")
    log.info(f"min_query_count {settings.min_query_count}")
    runner.start_run(result_dict, args.accuracy)

    lg.StartTestWithLogSettings(sut, qsl, settings, log_settings, audit_config)

    runner.finish()
    lg.DestroyQSL(qsl)
    lg.DestroySUT(sut)
    #post_proc.save_images(post_proc.content_ids[:10], ds)


if __name__ == "__main__":
    main()
