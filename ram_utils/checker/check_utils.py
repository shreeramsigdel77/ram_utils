import os

def file_directory_exist(path: str):
    try :
        os.path.exists(path)
    except FileExistsError as fe_error:
        print(fe_error)

def is_file(path: str):
    try:
        os.path.isfile(path)
    except FileNotFoundError as fnf_error:
        print(fnf_error)

def is_dir(path: str):
    try:
        os.path.isdir(path)
    except FileNotFoundError as fnf_error:
        print(fnf_error)