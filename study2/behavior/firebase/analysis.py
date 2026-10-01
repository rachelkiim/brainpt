import pandas as pd
import numpy as np
import os

HERE = os.path.dirname(os.path.abspath(__file__))

def load_data(file_path=os.path.join(HERE, '..', '..', '..', 'data', 'study2', 'behavior', 'GameResults_cleaned.xlsx')):
    """Load the game log."""
    df = pd.read_excel(file_path)
    return df

def classify_participants(df):
    """Classify participants as take-home, visit1 or visit2."""
    participants_info = {}

    for name in df['participantName'].unique():
        if '_exp1' in name:
            session_type = 'visit1'
            base_name = name.replace('_exp1', '')
        elif '_exp2' in name:
            session_type = 'visit2'
            base_name = name.replace('_exp2', '')
        else:
            session_type = 'take-home'
            base_name = name

        participants_info[name] = {
            'base_name': base_name,
            'session_type': session_type,
            'full_name': name
        }

    return participants_info

def count_cycles_takehome(df):
    """Number of completed cycles per take-home participant."""
    takehome_df = df[~df['participantName'].str.contains('_exp', na=False)]

    results = []

    for participant in takehome_df['participantName'].unique():
        participant_data = takehome_df[takehome_df['participantName'] == participant]

        # rows with iteration == 1 = completed cycles
        cycle_count = len(participant_data[participant_data['iteration'] == 1.0])

        results.append({
            '참가자': participant,
            'Cycle 시행 횟수': cycle_count,
            '총 레코드': len(participant_data)
        })

    return pd.DataFrame(results)

def calculate_game_time(df):
    """Time spent per game."""
    results = []

    for participant in df['participantName'].unique():
        participant_data = df[df['participantName'] == participant]

        for game in participant_data['gameName'].unique():
            game_data = participant_data[participant_data['gameName'] == game]

            if game_data['elapsedTime'].notna().sum() > 0:
                avg_time = game_data['elapsedTime'].mean()
                total_time = game_data['elapsedTime'].sum()
                count = len(game_data)

                results.append({
                    '참가자': participant,
                    '게임': game,
                    '평균 시간(초)': round(avg_time, 2),
                    '총 시간(초)': round(total_time, 2),
                    '플레이 횟수': count
                })

    return pd.DataFrame(results)

def calculate_cycle_time(df):
    """Time per cycle (sum over all games in one iteration)."""
    results = []

    for participant in df['participantName'].unique():
        participant_data = df[df['participantName'] == participant]

        for iteration in participant_data['iteration'].unique():
            if pd.notna(iteration):
                iteration_data = participant_data[participant_data['iteration'] == iteration]

                if iteration_data['elapsedTime'].notna().sum() > 0:
                    cycle_time = iteration_data['elapsedTime'].sum()
                    game_count = len(iteration_data)

                    results.append({
                        '참가자': participant,
                        'Iteration': int(iteration),
                        'Cycle 시간(초)': round(cycle_time, 2),
                        '게임 수': game_count
                    })

    result_df = pd.DataFrame(results)

    cycle1_df = result_df[result_df['Iteration'] == 1].copy()

    summary = cycle1_df.groupby('참가자').agg({
        'Cycle 시간(초)': ['mean', 'std', 'count']
    }).round(2)

    summary.columns = ['평균 Cycle 시간(초)', '표준편차', '시행 횟수']
    summary = summary.reset_index()

    return cycle1_df, summary

def calculate_accuracy_by_game(df):
    """Accuracy per game."""
    results = []

    for participant in df['participantName'].unique():
        participant_data = df[df['participantName'] == participant]

        for game in participant_data['gameName'].unique():
            game_data = participant_data[participant_data['gameName'] == game]

            if game_data['isCorrect'].notna().sum() > 0:
                accuracy = game_data['isCorrect'].mean()
                correct_count = game_data['isCorrect'].sum()
                total_count = game_data['isCorrect'].notna().sum()

                results.append({
                    '참가자': participant,
                    '게임': game,
                    '정답률': round(accuracy, 4),
                    '정답 수': int(correct_count),
                    '총 문제 수': total_count
                })

    return pd.DataFrame(results)

def main():
    print("=" * 80)
    print("게임 데이터 분석")
    print("=" * 80)

    df = load_data()
    print(f"\n데이터 로드 완료: {len(df)}개 레코드, {df['participantName'].nunique()}명")

    participants_info = classify_participants(df)
    takehome_count = sum(1 for p in participants_info.values() if p['session_type'] == 'take-home')
    visit1_count = sum(1 for p in participants_info.values() if p['session_type'] == 'visit1')
    visit2_count = sum(1 for p in participants_info.values() if p['session_type'] == 'visit2')

    print(f"\n참가자 구성:")
    print(f"  Take-home: {takehome_count}명")
    print(f"  Visit 1: {visit1_count}명")
    print(f"  Visit 2: {visit2_count}명")

    # 1. Cycles per take-home participant
    print("\n" + "=" * 80)
    print("1. Take-home 참가자의 1 Cycle 시행 횟수")
    print("=" * 80)
    cycle_counts = count_cycles_takehome(df)
    print(cycle_counts.to_string(index=False))

    # 2. Time per game
    print("\n" + "=" * 80)
    print("2. 각 게임별 걸리는 시간")
    print("=" * 80)
    game_times = calculate_game_time(df)
    print(game_times.head(20).to_string(index=False))
    print(f"\n... (총 {len(game_times)}개 결과)")

    # 3. Time per cycle
    print("\n" + "=" * 80)
    print("3. 1 Cycle 걸리는 시간")
    print("=" * 80)
    cycle_details, cycle_summary = calculate_cycle_time(df)
    print("\n참가자별 평균 Cycle 시간:")
    print(cycle_summary.to_string(index=False))

    # 4. Accuracy per game
    print("\n" + "=" * 80)
    print("4. 게임별 정답률")
    print("=" * 80)
    accuracy = calculate_accuracy_by_game(df)
    print(accuracy.head(20).to_string(index=False))
    print(f"\n... (총 {len(accuracy)}개 결과)")

    output_dir = os.path.join(HERE, 'analysis_results')
    os.makedirs(output_dir, exist_ok=True)

    cycle_counts.to_excel(f'{output_dir}/takehome_cycle_counts.xlsx', index=False)
    game_times.to_excel(f'{output_dir}/game_times.xlsx', index=False)
    cycle_details.to_excel(f'{output_dir}/cycle_times_detail.xlsx', index=False)
    cycle_summary.to_excel(f'{output_dir}/cycle_times_summary.xlsx', index=False)
    accuracy.to_excel(f'{output_dir}/game_accuracy.xlsx', index=False)

    print("\n" + "=" * 80)
    print(f"결과 파일 저장 완료: {output_dir}/")
    print("=" * 80)
    print("  - takehome_cycle_counts.xlsx")
    print("  - game_times.xlsx")
    print("  - cycle_times_detail.xlsx")
    print("  - cycle_times_summary.xlsx")
    print("  - game_accuracy.xlsx")

if __name__ == "__main__":
    main()
