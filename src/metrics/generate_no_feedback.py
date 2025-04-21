import os
import shutil

def rename_file(old_name, new_name):
    """Rename a file if it exists."""
    if os.path.exists(old_name):
        os.rename(old_name, new_name)
        print(f"Renamed '{old_name}' to '{new_name}'.")
    else:
        print(f"File '{old_name}' does not exist.")
        exit(1)

def move_file_to_parent(file_path):
    """Move file from its current folder to its parent directory."""
    if os.path.exists(file_path):
        # Determine the parent directory of the file's current folder.
        current_folder = os.path.dirname(file_path)  # e.g., 'feedback'
        parent_folder = os.path.dirname(os.path.abspath(current_folder))
        new_path = os.path.join(parent_folder, os.path.basename(file_path))
        shutil.move(file_path, new_path)
        print(f"Moved '{file_path}' to parent folder as '{new_path}'.")
        return new_path
    else:
        print(f"File '{file_path}' does not exist.")
        exit(1)

def remove_feedback_folder(folder_name):
    """Remove a folder and all its contents."""
    if os.path.exists(folder_name) and os.path.isdir(folder_name):
        try:
            shutil.rmtree(folder_name)
            print(f"Removed folder '{folder_name}' and its contents.")
        except Exception as e:
            print(f"Error removing folder '{folder_name}': {e}")
    else:
        print(f"Folder '{folder_name}' does not exist or is not a directory.")

def remove_file(file_path):
    """Remove a file if it exists."""
    if os.path.exists(file_path):
        try:
            os.remove(file_path)
            print(f"Removed existing file '{file_path}'.")
        except Exception as e:
            print(f"Error removing file '{file_path}': {e}")
            exit(1)

def main():
    folder = "./results_no_feedback_openai/generations/rplan/full_prompt/"
    for entry in os.listdir(folder):
        folder_path = os.path.join(folder, entry)
        if os.path.isdir(folder_path):
            feedback_folder = 'feedback'
            target_name = '0.json'
            old_filename = os.path.join(folder_path + "/" + feedback_folder, 'iteration_0.json')
            temp_filename = os.path.join(folder_path + "/" + feedback_folder, target_name)

            rename_file(old_filename, temp_filename)
            remove_file(target_name)
            move_file_to_parent(temp_filename)
            remove_feedback_folder(folder_path + "/" + feedback_folder)

if __name__ == '__main__':
    main()
