import pandas as pd

def split_csv(file_path):
    # Read the CSV file
    df = pd.read_csv(file_path)
    
    # Find the midpoint
    mid_index = len(df) // 2
    
    # Split the data into two halves
    df1 = df.iloc[:mid_index]
    df2 = df.iloc[mid_index:]
    
    # Save the two halves as new CSV files
    df1.to_csv("part1.csv", index=False)
    df2.to_csv("part2.csv", index=False)
    
    print("CSV file has been split into 'part1.csv' and 'part2.csv'.")

# Example usage
split_csv("/workspace/Inshallah/corrected_inertias_tests/trajectories/robot_0_trajectory.csv")
