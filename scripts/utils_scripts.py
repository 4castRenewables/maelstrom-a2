import logging

import a2.utils.file_handling


def _determine_path_output(args):
    path_output = f"{args.output_dir}/{args.task_name}/"
    logging.info(f".... using {path_output=}")
    a2.utils.file_handling.make_directories(path_output)
    return path_output
