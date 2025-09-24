import os
import pandas as pd
import random

def get_random_question(difficulty_level, csv_file_path="questions/question_1_personal.csv"):
    """
    Get a random question of specified difficulty level from CSV and remove it.
    
    Args:
        difficulty_level (int): Difficulty level (1, 2, or 3)
        csv_file_path (str): Path to the CSV file containing questions
        
    Returns:
        dict: Dictionary containing the selected question data, or None if no questions available
    """
    if difficulty_level not in [1, 2, 3]:
        raise ValueError("Difficulty level must be 1, 2, or 3")
    
    # Get absolute path relative to script location
    script_dir = os.path.dirname(os.path.abspath(__file__))
    full_csv_path = os.path.join(script_dir, "..", csv_file_path)
    
    if not os.path.exists(full_csv_path):
        raise FileNotFoundError(f"CSV file not found: {full_csv_path}")
    
    df = pd.read_csv(full_csv_path)
    
    filtered_questions = df[df['difficulty'] == difficulty_level]
    
    if filtered_questions.empty:
        return None
    
    selected_index = random.choice(filtered_questions.index)
    selected_question = df.loc[selected_index].to_dict()
    
    df = df.drop(selected_index)    
    df.to_csv(full_csv_path, index=False)
    
    return selected_question

def reset_csv_file(source_csv_path="questions/question_1.csv", target_csv_path="questions/question_1_personal.csv"):
    """
    Delete content from target CSV file and copy content from source CSV file into it.
    
    Args:
        source_csv_path (str): Path to the source CSV file to copy from
        target_csv_path (str): Path to the target CSV file to overwrite
        
    Raises:
        FileNotFoundError: If source CSV file doesn't exist
    """
    # Get absolute paths relative to script location
    script_dir = os.path.dirname(os.path.abspath(__file__))
    full_source_path = os.path.join(script_dir, "..", source_csv_path)
    full_target_path = os.path.join(script_dir, "..", target_csv_path)
    
    if not os.path.exists(full_source_path):
        raise FileNotFoundError(f"Source CSV file not found: {full_source_path}")
    
    df = pd.read_csv(full_source_path)    
    df.to_csv(full_target_path, index=False)
    
    def refresh_all_personal_csv_files():
        """
        Refresh all personal CSV files by copying from their corresponding source files.
        Looks for all CSV files in the questions directory and creates/refreshes their _personal versions.
        """
        script_dir = os.path.dirname(os.path.abspath(__file__))
        questions_dir = os.path.join(script_dir, "..", "questions")
        
        if not os.path.exists(questions_dir):
            raise FileNotFoundError(f"Questions directory not found: {questions_dir}")
        
        source_files = [f for f in os.listdir(questions_dir) if f.endswith('.csv') and not f.endswith('_personal.csv')]
        
        for source_file in source_files:
            base_name = source_file[:-4]  
            personal_file = f"{base_name}_personal.csv"
            
            source_path = f"questions/{source_file}"
            target_path = f"questions/{personal_file}"
            
            reset_csv_file(source_path, target_path)
            print(f"Refreshed: {personal_file}")
    