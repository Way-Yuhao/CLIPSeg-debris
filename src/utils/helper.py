import os

__author__ = 'Yuhao Liu'

def delete_dot_underscore_files(directory: str):
    """
    Delete all files that start with ._ in the specified directory and its subdirectories.

    Args:
    directory (str): The root directory in which to search for ._ files.
    """
    # Walk through the directory and all its subdirectories
    for root, dirs, files in os.walk(directory):
        for file in files:
            print(f"Found: {file}")
            # Check if the file starts with ._
            if file.startswith('._'):
                file_path = os.path.join(root, file)

                try:
                    os.remove(file_path)
                    print(f"Deleted: {file_path}")
                except Exception as e:
                    print(f"Failed to delete {file_path}: {e}")


if __name__ == "__main__":
    # Example usage
    delete_dot_underscore_files("/home/yuhaoliu/Data/HIDeAI/merged_multi_labeler/")