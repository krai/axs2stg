import logging

process_log = logging.getLogger("process_logger")
process_log.setLevel(logging.INFO)
process_handler = logging.FileHandler('process_log.csv')
process_handler.setFormatter(logging.Formatter('%(message)s'))
process_log.addHandler(process_handler)

def log_event(gpu_id, process_type, timestamp_start, timestamp_end):
    process_log.info(f"{gpu_id},{process_type},{timestamp_start},{timestamp_end}")
