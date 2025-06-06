from datetime import datetime
import os


def create_file(figure_prefix=""):
    file = figure_prefix+"test.txt"
    print("Path:", file)
    with open(file, 'w') as f:
        f.write("ayo")
    
    print("Success")



if __name__ == "__main__":
    current_time = datetime.now().strftime("%m-%d/%H-%M-%S")  # Month-Day Hour-Minute
    time_folder = "exports/"+current_time
    os.makedirs(time_folder, exist_ok=True)

    create_file(figure_prefix=time_folder+"/")