

import pandas as pd
import glob
import textgrids
import re
files_pred = glob.glob(f'data/respinpeace/cv_pred_*.TextGrid')
files_gt = glob.glob(f'data/respinpeace_gt/*.TextGrid')

audible_intersection = []
for overlap_threshold in [0.7, 0.8, 0.9]:
    for file_path in files_pred:
        match = re.search(r'cv_pred_(.*?)\.TextGrid', file_path)
        if match:
            sample_id = match.group(1)
        intervals_pred = textgrids.TextGrid(file_path)
        intervals_gt = textgrids.TextGrid(f'data/respinpeace_gt/gt_{sample_id}.TextGrid')

        audible_breath = pd.read_csv(f'data/respir-en_predictions/breath_prediction_{sample_id}.csv')
        audible_breath = audible_breath.sort_values('start')
        
        for intervals, tag in zip([intervals_pred, intervals_gt], ['pred', 'gt']):
            inhale_intervals = []
            for interval in intervals['resp']:
                if interval.text:
                    if 'in' in interval.text.lower():
                        inhale_start = interval.xmin
                        inhale_end = interval.xmax 
                        inhale_intervals.append((inhale_start, inhale_end))
            
            # Get audible breath intervals
            audible_intervals = [(row.start, row.end) for row in audible_breath.itertuples()]
            
            intersection_count = 0
            for aud_start, aud_end in audible_intervals:
                aud_duration = aud_end - aud_start
                
                for inh_start, inh_end in inhale_intervals:
                    # Calculate overlap between audible breath and inhale event
                    overlap_start = max(aud_start, inh_start)
                    overlap_end = min(aud_end, inh_end)
                    overlap_duration = max(0, overlap_end - overlap_start)
                    
                    # Calculate overlap percentage relative to audible breath duration
                    overlap_percentage = overlap_duration / aud_duration if aud_duration > 0 else 0
                    
                    if overlap_percentage >= overlap_threshold:
                        intersection_count += 1
                        break  # Move to next audible interval after first match

            audible_intersection.append({'sample_id': sample_id, 'dataset': tag, 'total_audible': len(audible_intervals),
                    'total_inhale_events': len(inhale_intervals),
                    'intersection': intersection_count,
                    'intersection_percentage': (intersection_count / len(audible_intervals) * 100) if audible_intervals else 0,
                    'overlap_threshold': overlap_threshold
                })

audible_intersection = pd.DataFrame(audible_intersection)
audible_intersection.to_csv('intersection.csv')