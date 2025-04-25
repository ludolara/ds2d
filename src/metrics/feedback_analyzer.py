import os
import glob
import json
import pandas as pd
import matplotlib.pyplot as plt

class FeedbackAnalyzer:
    def __init__(self, feedback_file=None, feedback_dir=None):
        feedback_list = []

        if feedback_dir:
            pattern = os.path.join(feedback_dir, "**", "feedback", "*.json")
            for path in glob.glob(pattern, recursive=True):
                with open(path, 'r') as f:
                    data = json.load(f)
                if isinstance(data, list):
                    feedback_list.extend(data)
                else:
                    feedback_list.append(data)

        if feedback_file:
            with open(feedback_file, 'r') as f:
                data = json.load(f)
            if isinstance(data, list):
                feedback_list.extend(data)
            else:
                feedback_list.append(data)

        if not feedback_list:
            raise ValueError("No feedback found: provide a valid feedback_file or feedback_dir")
        
        self.feedback = feedback_list
        self.df = self._build_dataframe()

    def _build_dataframe(self):
        records = []
        for entry in self.feedback:
            it = entry.get('iteration')
            m = entry.get('metrics', {})
            rc = m.get('room_count', {})
            records.append({
                'iteration': it,
                'is_overlapping': m.get('is_overlapping'),
                'total_overlap_area': m.get('total_overlap_area'),
                'overlap_percentage': m.get('overlap_percentage'),
                'is_valid_json': m.get('is_valid_json'),
                'room_count_actual': rc.get('actual'),
                'room_count_match': rc.get('match'),
                'total_area_match': m.get('total_area', {}).get('match'),
            })
        return pd.DataFrame(records)
    
    def plot_overlap_status_by_room_count(self, iteration):
        # if iteration == -1:
        #     idxs = self.df.groupby('feedback')['iteration'].idxmax()
        #     df_it = self.df.loc[idxs]
        #     title_iter = "last"
        # else:
        df_it = self.df[self.df['iteration'] == iteration]

        ok = df_it[(df_it['is_overlapping'] == False) & (df_it['is_valid_json'] == True)]
        bad = df_it[df_it['is_overlapping'] == True]

        ok_counts  = ok['room_count_actual'].value_counts().sort_index()
        bad_counts = bad['room_count_actual'].value_counts().sort_index()
        
        global_max = int(ok.max().max()) + 1

        counts = pd.DataFrame({
            'Non-overlapping': ok_counts,
            'Overlapping'    : bad_counts
        }).fillna(0)

        ax = counts.plot(
            kind='bar',
            color=['#4caf50', '#f44336'], 
            figsize=(8,5)
        )
        ax.set_xlim(0.5, 6)
        ax.set_xlabel('Room Count')
        ax.set_ylabel('Number of fp')
        ax.set_title(f'Iteration {iteration}: Overlap Status by Room Count')
        ax.legend()
        plt.tight_layout()
        plt.savefig(f"out_{iteration}.png", dpi=300)
        plt.close()

analyzer = FeedbackAnalyzer(feedback_dir='results/generations/no_doors_rplan_20_70B/full_prompt/')
analyzer.plot_overlap_status_by_room_count(0) 
