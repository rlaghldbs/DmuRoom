import streamlit as st
import pandas as pd
import re
from itertools import combinations
from io import BytesIO

# --------------------------------------------------------------------------
# [1] 기존 배정 로직 (Backend) - print()문을 st.write()나 로그로 변경
# --------------------------------------------------------------------------

# 데이터 정의는 코드에 유지

# 학과 -> 학부 매핑
MAJOR_TO_FACULTY_MAP = {
    '기계공학과': '기계공학부', '기계설계공학과': '기계공학부', '자동화공학과': '로봇자동화공학부', '로봇소프트웨어과': '로봇자동화공학부',
    '전기공학과': '전기전자통신공학부', '반도체전자공학과': '전기전자통신공학부', '정보통신공학과': '전기전자통신공학부', '소방안전관리과': '전기전자통신공학부',
    '웹응용소프트웨어공학과': '컴퓨터공학부', '컴퓨터소프트웨어공학과': '컴퓨터공학부', '인공지능소프트웨어학과': '컴퓨터공학부', '생명화학공학과': '생활환경공학부',
    '바이오융합공학과': '생활환경공학부', '건축과': '생활환경공학부', '실내건축디자인과': '생활환경공학부', '시각디자인과': '생활환경공학부',
    'AR·VR콘텐츠디자인과': '생활환경공학부', '경영학과': '경영학부', '세무회계학과': '경영학부', '유통마케팅학과': '경영학부',
    '호텔관광학과': '경영학부', '경영정보학과': '경영학부', '빅데이터경영과': '경영학부', '자유전공학과': '자유전공학부'
}
# 기숙사 실명 긴 형태 -> 짧은 형태 매핑
DORM_LONG_TO_SHORT_MAP = {
    'A형(기숙사형 2인호의 2인실)': 'A형', 'B형(기숙사형 2인호의 1인실)': 'B형',
    'C형(기숙사형 3인호의 1인실)': 'C형', 'D형(기숙사형 3인호의 2인실)': 'D형',
    'E형(기숙사형 4인호의 2인실)': 'E형', 'F형(아파트형 1인실(여학생 전용))': 'F형',
    'G형(아파트형 2인실(여학생 전용))': 'G형'
}

#방설정 파일
def load_room_config(config_file):
    logs = []
    config_df = None
    try:
        config_df = pd.read_excel(config_file, dtype=str)
        config_df['room'] = pd.to_numeric(config_df['room'])
        config_df['amount'] = pd.to_numeric(config_df['amount'])
    except Exception as e:
        logs.append(f"🚨 오류: 기숙사 방 정보 파일 처리 중 오류 발생: {e}")
        return None, None, None, None,logs
    
    available_rooms, room_capacities, room_prices = {}, {}, {}
    
    for name, group in config_df.groupby('Type'):
        room_capacities[name] = group['room'].iloc[0]
        room_prices[name] = group['amount'].iloc[0]
        
        gender_rooms = {}
        for gender, sub_group in group.groupby('sex'):
            gender_rooms[gender] = sorted(sub_group['Room_No'].unique().tolist())
        available_rooms[name] = gender_rooms
    logs.append("기숙사 방 정보 파일을 성공적으로 읽었습니다.")
    return available_rooms, room_capacities, room_prices, config_df, logs

def find_best_pair_info(unassigned_students):
    """학생 그룹 내에서 최적의 짝을 찾아 정보를 반환하는 헬퍼 함수"""
    possible_pairs = []
    student_tuples = list(unassigned_students.itertuples(index=True))
    for s1, s2 in combinations(student_tuples, 2):
        score, reasons = 0, []
        is_same_smoking = (s1.흡연여부 == s2.흡연여부)
        is_same_major = (s1.학과 == s2.학과)
        is_same_faculty = (hasattr(s1, '학부') and hasattr(s2, '학부') and s1.학부 == s2.학부)
        is_same_pattern=(s1.생활패턴==s2.생활패턴)
        if is_same_smoking:
            if is_same_major:
                score = 10; reasons = ['흡연 여부 동일', '동일 학과']
                if is_same_pattern: score +=2; reasons.append('생활패턴 동일')
                
            elif is_same_faculty:
                 score = 8; reasons = ['흡연 여부 동일', '동일 학부']
                 if is_same_pattern: score +=2; reasons.append('생활패턴 동일')
            else: 
                score = 6; reasons = ['흡연 여부 동일']
                if is_same_pattern: score +=2; reasons.append('생활패턴 동일')
        else:
            if is_same_major:
                 score = 4; reasons = ['혼합 배정 (동일 학과)']
                 if is_same_pattern: score +=2; reasons.append('생활패턴 동일')
                 
            elif is_same_faculty:
                 score = 2; reasons = ['혼합 배정 (동일 학부)']
                 if is_same_pattern: score +=2; reasons.append('생활패턴 동일')
        if score > 0:
            possible_pairs.append({'pair': (s1.Index, s2.Index), 'score': score, 'reason': ', '.join(reasons)})

    if not possible_pairs:
        if len(unassigned_students) >= 2:
            return {'pair': (unassigned_students.index[0], unassigned_students.index[1]), 'reason': '랜덤 배정'}
        return None

    best_match_map = {s.Index: (-1, None) for s in student_tuples}
    for pair in possible_pairs:
        p1_idx, p2_idx = pair['pair']; score = pair['score']
        if score > best_match_map[p1_idx][0]: best_match_map[p1_idx] = (score, p2_idx)
        if score > best_match_map[p2_idx][0]: best_match_map[p2_idx] = (score, p1_idx)
    
    mutual_best_pairs = []
    processed = set()
    for s1_idx, (score, s2_idx) in best_match_map.items():
        if s1_idx in processed or s2_idx is None: continue
        if best_match_map.get(s2_idx, (None, None))[1] == s1_idx:
            reason = [p['reason'] for p in possible_pairs if set(p['pair']) == {s1_idx, s2_idx}][0]
            mutual_best_pairs.append({'pair': (s1_idx, s2_idx), 'score': score, 'reason': reason})
            processed.add(s1_idx); processed.add(s2_idx)
    
    if mutual_best_pairs:
        mutual_best_pairs.sort(key=lambda x: x['score'], reverse=True)
        return mutual_best_pairs[0]
    else:
        possible_pairs.sort(key=lambda x: x['score'], reverse=True)
        return possible_pairs[0]

def assign_dorm_rooms(student_file, available_rooms, room_capacities):
    logs = []
    try:
        df = pd.read_excel(student_file, dtype=str)
    except Exception as e:
        logs.append(f"🚨 오류: 학생 데이터 파일 처리 중 오류 발생: {e}")
        return None, 0, 0, logs

    initial_count = len(df)
    logs.append(f"✅ [1단계] 엑셀 파일에서 총 {initial_count}명의 학생을 읽었습니다.")
    
    # (이하 배정 로직은 이전과 동일, print 문 대신 logs 리스트에 추가)
    df['기숙사 실'] = df['기숙사 실'].str.strip()
    df['타입'] = df['기숙사 실'].map(DORM_LONG_TO_SHORT_MAP)
    df.rename(columns={'학과(필수)': '학과', '희망하는 룸메이트 기재': '희망룸메이트'}, inplace=True)
    df['학부'] = df['학과'].map(MAJOR_TO_FACULTY_MAP)
    
    defined_types = set(room_capacities.keys())
    unmatched_df = df[~df['타입'].isin(defined_types) | df['타입'].isna()]
    unmatched_count = len(unmatched_df)
    if not unmatched_df.empty:
        logs.append("---")
        logs.append("🚨 **경고**: 처리할 수 없는 '기숙사 실' 값을 가진 학생이 있습니다. 배정에서 제외됩니다.")
        # 로그에는 상위 5명만 보여줌
        logs.append(unmatched_df[['성명', '학번', '기숙사 실']].head().to_string())
        logs.append("---")
    df = df[df['타입'].isin(defined_types)]
    
    final_assignments = []
    for (dorm_type, gender), group in df.groupby(['타입', '성별']):
        unassigned_students = group.copy()
        rooms = available_rooms.get(dorm_type, {}).get(gender, []).copy()
        capacity = room_capacities.get(dorm_type, 2)

        if dorm_type == 'B형':
            room_pairs = []
            for i in range(0, len(rooms), 2):
                if i + 1 < len(rooms) and rooms[i][:-1] == rooms[i+1][:-1]:
                    room_pairs.append((rooms[i], rooms[i+1]))
            
            while len(unassigned_students) >= 2:
                if not room_pairs: break
                target_pair_info = find_best_pair_info(unassigned_students)
                if not target_pair_info: break
                
                idx1, idx2 = target_pair_info['pair']
                room1, room2 = room_pairs.pop(0)
                reason = target_pair_info['reason']

                s1_info = unassigned_students.loc[idx1].to_dict()
                s2_info = unassigned_students.loc[idx2].to_dict()
                s1_info.update({'방 번호': room1, '선정 이유': reason})
                s2_info.update({'방 번호': room2, '선정 이유': reason})
                final_assignments.extend([s1_info, s2_info])
                unassigned_students.drop(index=[idx1, idx2], inplace=True)

            if not unassigned_students.empty:
                s_info = unassigned_students.iloc[0].to_dict()
                s_info.update({'방 번호': '배정 보류', '선정 이유': '최종 잔여 인원 (B형)'})
                final_assignments.append(s_info)                
            continue

        if capacity == 1:
            for _, student in unassigned_students.iterrows():
                if not rooms:
                    s_info = student.to_dict()
                    s_info.update({'방 번호': '배정 보류', '선정 이유': '1인실 부족'})
                    final_assignments.append(s_info)
                    continue
                s_info = student.to_dict()
                s_info.update({'방 번호': rooms.pop(0), '선정 이유': '1인실 배정'})
                final_assignments.append(s_info)
            print(f"--- 그룹 처리 완료: [{dorm_type} / {gender}] ---")
            continue
        
        assigned_indices = set()
        unassigned_students.drop(index=list(assigned_indices), errors='ignore', inplace=True)

        for idx, student in unassigned_students.iterrows():
            if idx in assigned_indices: continue
            raw_request = student['희망룸메이트']
            if pd.isna(raw_request) or str(raw_request).strip() == '': continue
            match = re.match(r'\d+', str(raw_request).strip())
            if not match: continue
            requested_id = match.group(0)
            roommate_df = unassigned_students[(unassigned_students['학번'] == requested_id) & (~unassigned_students.index.isin(assigned_indices))]
            if not roommate_df.empty:
                roommate = roommate_df.iloc[0]
                if not pd.isna(roommate['희망룸메이트']):
                    roommate_match = re.match(r'\d+', str(roommate['희망룸메이트']).strip())
                    if roommate_match and roommate_match.group(0) == student['학번']:
                        if not rooms: break
                        room_num = rooms.pop(0)
                        for r_idx in [idx, roommate.name]:
                            s_info = unassigned_students.loc[r_idx].to_dict()
                            s_info.update({'방 번호': room_num, '선정 이유': '상호 희망'})
                            final_assignments.append(s_info)
                            assigned_indices.add(r_idx)
        unassigned_students.drop(index=list(assigned_indices), errors='ignore', inplace=True)

        iteration = 1
        while len(unassigned_students) >= 2:
            if not rooms:
                break
            target_pair_info = find_best_pair_info(unassigned_students)
            if not target_pair_info:
                break
            idx1, idx2 = target_pair_info['pair']
            room_num = rooms.pop(0)
            for r_idx in [idx1, idx2]:
                s_info = unassigned_students.loc[r_idx].to_dict()
                s_info.update({'방 번호': room_num, '선정 이유': target_pair_info['reason']})
                final_assignments.append(s_info)
            unassigned_students.drop(index=[idx1, idx2], inplace=True)

        if not unassigned_students.empty:
            s_info = unassigned_students.iloc[0].to_dict()
            s_info.update({'방 번호': '배정 보류', '선정 이유': '최종 잔여 인원'})
            final_assignments.append(s_info)

        print(f"--- 그룹 처리 완료: [{dorm_type} / {gender}] ---")
    
    return pd.DataFrame(final_assignments), initial_count, unmatched_count, logs

# --------------------------------------------------------------------------
# [2] Streamlit 웹 UI (Frontend)
# --------------------------------------------------------------------------

st.set_page_config(page_title="기숙사 자동 배정 시스템", layout="wide")

st.title("👨‍👩‍👧‍👦 기숙사 자동 배정 프로그램")
st.write("---")

# 파일 업로드 UI
st.header("📄 1. 파일 업로드")
st.info("학생 데이터와 기숙사 방 정보 엑셀 파일을 각각 업로드해주세요.")

col1, col2 = st.columns(2)
with col1:
    student_file = st.file_uploader("**학생 데이터 엑셀 파일 (students_data.xlsx)**", type=['xlsx'])
with col2:
    room_config_file = st.file_uploader("**기숙사 방 정보 엑셀 파일 (room_config.xlsx)**", type=['xlsx'])

st.write("---")

# 배정 실행 버튼
if st.button("🚀 배정 실행하기", type="primary"):
    if student_file is not None and room_config_file is not None:
        # 1. 기숙사 방 정보 로드
        with st.spinner('STEP 1/3: 기숙사 방 정보를 읽는 중...'):
            available_rooms, room_capacities, room_prices, config_df, config_logs = load_room_config(room_config_file)
            for log in config_logs:
                st.write(log)
        
        if available_rooms:
            st.header("💰 납부금액 확인")
            with st.spinner('학생들의 납부금액을 확인하는 중...'):
                student_df_for_check = pd.read_excel(student_file, dtype=str)
                student_df_for_check['타입'] = student_df_for_check['기숙사 실'].map(DORM_LONG_TO_SHORT_MAP)
                student_df_for_check['정상금액'] = student_df_for_check['타입'].map(room_prices)

                # 납부금액과 정상금액을 숫자로 변환 (오류 발생 시 NaN으로 처리)
                student_df_for_check['납부금액'] = pd.to_numeric(student_df_for_check['납부금액'], errors='coerce')
                student_df_for_check['정상금액'] = pd.to_numeric(student_df_for_check['정상금액'], errors='coerce')

                # 금액이 다른 학생들 필터링
                mismatched_payments = student_df_for_check[
                    student_df_for_check['납부금액'] != student_df_for_check['정상금액']
                ].dropna(subset=['납부금액', '정상금액'])

                if mismatched_payments.empty:
                    st.success("✅ 모든 학생의 납부금액이 정상적으로 확인되었습니다.")
                else:
                    st.error(f"🚨 {len(mismatched_payments)}명의 학생에게서 납부금액 불일치가 발견되었습니다.")
                    st.dataframe(mismatched_payments[['성명', '학번', '기숙사 실', '납부금액', '정상금액']])
            st.write("---")
            # 2. 메인 배정 로직 실행
            with st.spinner('STEP 2/3: 최적의 룸메이트를 찾고 있습니다... (학생 수가 많으면 몇 분 정도 소요될 수 있습니다)'):
                assignments_df, initial_count, unmatched_count, assign_logs = assign_dorm_rooms(student_file, available_rooms, room_capacities)
                for log in assign_logs:
                    st.write(log)
            
            # 3. 최종 결과 출력
            with st.spinner('STEP 3/3: 최종 결과 파일을 생성하는 중...'):
                st.header("📊 2. 최종 배정 결과")

                if config_df is not None:
                    st.info("배정되지 않은 빈 방(공실)을 최종 결과에 추가합니다...")
                    
                    all_rooms_df = config_df.rename(columns={'Room_No': '방 번호', 'Type': '타입', 'sex': '성별', 'Max': 'Max'})
                    assigned_rooms = set(assignments_df['방 번호'])
                    vacant_rooms_df = all_rooms_df[~all_rooms_df['방 번호'].isin(assigned_rooms)].copy()

                    if not vacant_rooms_df.empty:
                        new_vacant_rows = []
                        short_to_long_map = {v: k for k, v in DORM_LONG_TO_SHORT_MAP.items()}
                    
                        for _, room_info in vacant_rooms_df.iterrows():
                            capacity = int(room_info['Max'])
                            
                            base_row = {
                                '기숙사 실': short_to_long_map.get(room_info['타입']),
                                '타입': room_info['타입'],
                                '방 번호': room_info['방 번호'],
                                '성별': room_info['성별'],
                                '선정 이유': '공실'
                            }
                            
                            # 해당 방의 정원(capacity)만큼 '공실' 행을 추가
                            for _ in range(capacity):
                                new_vacant_rows.append(base_row.copy())
                        
                        if new_vacant_rows:
                            vacant_df = pd.DataFrame(new_vacant_rows)
                            assignments_df = pd.concat([assignments_df, vacant_df], ignore_index=True)
        
                final_count = len(assignments_df)
                
                st.write(f"✅ 최종적으로 **{final_count}명**의 학생이 배정 결과에 포함되었습니다.")
                st.success("🟢 모든 학생이 정상적으로 처리되었습니다.")

                if not assignments_df.empty:
                    # 최종 DataFrame 가공
                    assignments_df['타입'] = assignments_df['기숙사 실'].map(DORM_LONG_TO_SHORT_MAP)
                    assignments_df['호실'] = assignments_df['방 번호'].str[:3]
                    assignments_df.rename(columns={'학과': '학과(필수)', '희망룸메이트': '희망하는 룸메이트 기재'}, inplace=True)
                    column_order = [
                        '기숙사 실', '타입', '방 번호', '호실', '성별', 
                        '학부', '학과(필수)', '학번', '성명', '본인 핸드폰 번호', '흡연여부','생활패턴',
                        '희망하는 룸메이트 기재','금액', '선정 이유'
                    ]
                    final_df = assignments_df.reindex(columns=column_order).sort_values(
                        by=['기숙사 실', '방 번호', '학번']
                    ).reset_index(drop=True)
                    final_df['금액'] = pd.to_numeric(final_df['금액'])

                    st.dataframe(final_df)

                    # 다운로드 버튼 생성
                    output = BytesIO()
                    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                        final_df.to_excel(writer, index=False, sheet_name='배정 결과')
                    
                    st.download_button(
                        label="📥 결과 엑셀 파일 다운로드",
                        data=output.getvalue(),
                        file_name="방배정_완료.xlsx",
                        mime="application/vnd.ms-excel"
                    )
    else:
        st.error("🚨 학생 데이터와 기숙사 방 정보 파일을 모두 업로드해야 합니다.")