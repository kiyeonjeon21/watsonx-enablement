"""
Create sample dataset for Milvus example
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import random

def create_sample_data(output_path: str, num_records: int = 100):
    """Create a sample dataset of news dialogues"""
    
    # Sample programs
    programs = ['Morning Edition', 'All Things Considered', 'Weekend Edition', 'Fresh Air']
    
    # Sample topics for generating titles and content
    topics = [
        ('Climate Change', [
            'Rising global temperatures impact local communities',
            'New research reveals accelerating ice melt in Antarctica',
            'Renewable energy adoption rates exceed expectations'
        ]),
        ('Technology', [
            'AI developments raise ethical questions',
            'New quantum computing breakthrough announced',
            'Tech giants face regulatory challenges'
        ]),
        ('Healthcare', [
            'Medical researchers announce breakthrough treatment',
            'Healthcare accessibility remains a challenge',
            'New study reveals impact of lifestyle on longevity'
        ]),
        ('Education', [
            'Remote learning transforms education landscape',
            'New teaching methodologies show promising results',
            'Education funding debates continue'
        ])
    ]
    
    # Generate random data
    data = []
    base_date = datetime.now() - timedelta(days=365)
    
    for i in range(num_records):
        topic, contents = random.choice(topics)
        program = random.choice(programs)
        date = base_date + timedelta(days=random.randint(0, 364))
        
        record = {
            'episode_id': f'EP{i+1:04d}',
            'program': program,
            'title': f'{topic}: {random.choice(contents)}',
            'episode_date': date.strftime('%Y-%m-%d'),
            'dialogue': f"""
                Today on {program}, we're discussing {topic.lower()}.
                {random.choice(contents)}.
                Our experts weigh in on the implications and future developments.
                This is a critical issue that affects communities worldwide.
                We'll continue to follow this story and bring you updates as they develop.
                Back to you in the studio.
            """.strip()
        }
        data.append(record)
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Save to CSV
    df.to_csv(output_path, index=False)
    print(f"Created sample dataset with {num_records} records at: {output_path}")

if __name__ == "__main__":
    output_path = "data/sample_dialogue_data.csv"
    create_sample_data(output_path, num_records=100)