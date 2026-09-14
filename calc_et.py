import argparse
from configparser import ConfigParser
from glob import glob
import datetime as dt
import pandas as pd
import numpy as np
from pandas.tseries.offsets import MonthEnd
import refinitiv.data as rd


# ==============================================================================
# 1. 환경 설정 및 세션 관리 모듈
# ==============================================================================
def load_config(config_path: str = 'D:/config.ini') -> str:
    """설정 파일에서 Refinitiv API 키를 로드합니다."""
    config = ConfigParser()
    config.read(config_path)
    return config['main']['api_key']


def open_refinitiv_session(app_key: str):
    """Refinitiv 세션을 활성화합니다."""
    session = rd.session.desktop.Definition(app_key=app_key).get_session()
    rd.session.set_default(session)
    session.open()


# ==============================================================================
# 2. 유틸리티 및 시계열 변환 매핑 모듈
# ==============================================================================
def add_quarter(curr_fq: str, adder: int) -> str:
    """분기 문자열(예: '2Q26AS')에 n분기를 더해 반환합니다."""
    while adder > 0:
        q = curr_fq[0]
        y = curr_fq[2:4]
        if q == '4':
            curr_fq = f"1Q{int(y) + 1:02d}AS"
        else:
            curr_fq = f"{int(q) + 1}Q{y}AS"
        adder -= 1
    return curr_fq


def get_FQFY(date_series: pd.Series) -> pd.Series:
    """날짜 데이터를 기반으로 현재 Fiscal Quarter/Year 규칙을 벡터로 맵핑합니다."""
    month = date_series.dt.month
    year_int = (date_series.dt.year % 100).fillna(0).astype(int)
    year_prev_int = ((date_series.dt.year - 1) % 100).fillna(0).astype(int)

    year_str = year_int.astype(str).str.zfill(2)
    year_prev_str = year_prev_int.astype(str).str.zfill(2)

    conditions = [
        month == 1,
        month.isin([2, 3, 4]),
        month.isin([5, 6, 7]),
        month.isin([8, 9, 10]),
        month.isin([11, 12])
    ]
    # f-string 대신 판다스 고속 컬럼 결합(+) 연산 사용
    choices = [
        "4Q" + year_prev_str + "AS",
        "1Q" + year_str + "AS",
        "2Q" + year_str + "AS",
        "3Q" + year_str + "AS",
        "4Q" + year_str + "AS"
    ]
    # 일치하는 패턴이 없으면 None 처리 (지정되지 않은 레이아웃 대비)
    return pd.Series(np.select(conditions, choices, default=None), index=date_series.index)


def get_next_FQFY(date_series: pd.Series) -> pd.Series:
    """날짜 데이터를 기반으로 차기 Fiscal Quarter/Year 규칙을 벡터로 맵핑합니다."""
    month = date_series.dt.month

    # 💡 마찬가지로 소수점 제거 후 문자열 패딩 처리
    year_int = (date_series.dt.year % 100).fillna(0).astype(int)
    year_next_int = ((date_series.dt.year + 1) % 100).fillna(0).astype(int)

    year_str = year_int.astype(str).str.zfill(2)
    year_next_str = year_next_int.astype(str).str.zfill(2)

    conditions = [
        month == 1,
        month.isin([2, 3, 4]),
        month.isin([5, 6, 7]),
        month.isin([8, 9, 10]),
        month.isin([11, 12])
    ]

    choices = [
        "1Q" + year_str + "AS",
        "2Q" + year_str + "AS",
        "3Q" + year_str + "AS",
        "4Q" + year_str + "AS",
        "1Q" + year_next_str + "AS"
    ]
    return pd.Series(np.select(conditions, choices, default=None), index=date_series.index)


def formatted_growth_rate(x, x_1):
    if x > 0 and x_1 > 0:
        return (x / x_1 - 1) * 100
    elif x_1 > 0 and x < 0:
        return 'T/L'
    elif x_1 < 0 and x < 0:
        return 'R/L'
    elif x_1 < 0 and x > 0:
        return 'T/P'
    else:
        return None


# ==============================================================================
# 3. 핵심 리포트 프로세싱 파이프라인 모듈
# ==============================================================================
def process_earning_tracker(country: str, q_basis: str, current_quarter: str, loc=-1) -> list:
    """Earning Tracker 리포트용 시트 데이터 리스트를 생성합니다."""
    print('Preparing Earning Tracker Report...')

    mktname = 'All(Top Free-Float Mkt Cap 500)' if country == 'us' else 'All(Coverage Analyst >= 3)'

    result_list = glob(f"result/{country}/mixed_model_{q_basis}_Q_sector_*.csv")
    if not result_list:
        raise FileNotFoundError(f"No sector files found for country: {country}")
    result_list.sort()

    recent_q = pd.read_csv(result_list[loc])
    recent_q.loc[recent_q['Sector'].isin([0, '0', '00']), 'Sector_name'] = mktname

    # -------------------------------------------------------------------------
    # OPTIMIZATION: O(N^2) 행 반복 조회를 Self-Merge 조인 연산으로 대체
    # -------------------------------------------------------------------------
    recent_q['q_basis_next'] = recent_q[q_basis] + 1

    lookup_df = recent_q[['Sector', 'FY', q_basis, 'earning_G_bld', 'earning_total_bld']].rename(
        columns={q_basis: 'q_basis_next', 'earning_G_bld': 'earning_G_bld_prev', 'earning_total_bld': 'earning_total_bld_prev'}
    )
    recent_q = pd.merge(recent_q, lookup_df, on=['Sector', 'FY', 'q_basis_next'], how='left')
    recent_q.drop('q_basis_next', axis=1, inplace=True)

    recent_q['chg'] = recent_q.apply(lambda x: formatted_growth_rate(x['earning_total_bld'], x['earning_total_bld_prev']), axis=1)
    recent_q['surp'] = recent_q.apply(lambda x: formatted_growth_rate(x['earning_total_bld'], x['earning_EW_total_bld']), axis=1)

    final_result = []
    for fq in range(4):
        curr_fy = add_quarter(current_quarter, fq)
        df_fq = recent_q[recent_q['FY'] == curr_fy]

        if df_fq.empty:
            final_result.append(pd.DataFrame())
            continue

        min_eqbtw = df_fq[q_basis].min()
        df_fq_min_eqbtw = df_fq[df_fq[q_basis] == min_eqbtw].copy()

        cols = ['Sector_name', 'Sector', 'FY', q_basis, 'earning_G_bld_prev', 'earning_G_bld', 'earning_EW_G_bld',
                'surp', 'chg']
        df_fq_result = df_fq_min_eqbtw[cols].astype({
            'Sector': pd.StringDtype(), 'earning_G_bld': float, 'earning_G_bld_prev': float,
            'earning_EW_G_bld': float, 'chg': float, 'surp': float
        }, errors='ignore')

        df_fq_result['Sector_len'] = df_fq_result['Sector'].str.len()
        df_fq_result = df_fq_result.sort_values(by=['Sector_len', 'Sector']).drop('Sector_len', axis=1)

        final_result.append(df_fq_result)
        print(f'Processing done for {curr_fy}...')

    return final_result


def process_expected_equity(country: str, q_basis: str, sector_len: int) -> pd.DataFrame:
    """Refinitiv 오픈 API 인터페이스 및 결측치 롤링 매핑을 처리하여 실적 발표 예정 데이터를 가공합니다."""
    print('Preparing Expected Report Equity Data Sheet...')

    equity_data_list = glob(f"result/{country}/mixed_model_Q_*.csv")
    if not equity_data_list:
        raise FileNotFoundError(f"No equity files found for country: {country}")
    equity_data_list.sort()

    equity_data = pd.read_csv(equity_data_list[-1], index_col=0).astype({'Sector': pd.StringDtype()})

    # 인더스트리 마스터 매핑
    ind = pd.read_excel(f'data/{country}/infos.xlsx', sheet_name='industry_map', dtype=str).set_index('Code')
    equity_data['SectorCode'] = equity_data['Sector'].str[:sector_len]
    equity_data['GICS Sector'] = equity_data['SectorCode'].map(ind['Sector'])
    equity_data['GICS Industry'] = equity_data['Sector'].map(ind['Sector'])

    codes = equity_data['Code'].unique().tolist()
    if country == 'kr':
        isin = pd.read_excel(f'data/{country}/infos.xlsx', sheet_name='isin', index_col=0, dtype=str)
        ds_codes = [isin.loc[k][0] for k in codes]
    else:
        ds_codes = codes

    print('Fetching Expected Report Dates and Ticker Symbols from Refinitiv...')
    exprepdate = rd.get_data(
        universe=ds_codes,
        fields=['TR.ExpectedReportDate', 'TR.ExpectedReportDate.periodenddate', 'TR.TickerSymbol'],
        parameters={'period': 'FQ1'}
    ).set_index('Instrument').rename(columns={'Period End Date': 'Expected Period End Date'})

    # 인덱스 유실 방지 및 고속 벡터 보정
    exprepdate['Expected Report Date'] = pd.to_datetime(exprepdate['Expected Report Date'])
    exprepdate['Expected Period End Date'] = pd.to_datetime(exprepdate['Expected Period End Date'])

    date_diff_mask = (exprepdate['Expected Report Date'] - exprepdate['Expected Period End Date']) > pd.Timedelta(
        days=90)
    exprepdate['Expected Period End Date'] = np.where(
        date_diff_mask,
        exprepdate['Expected Period End Date'] + pd.Timedelta(days=90),
        exprepdate['Expected Period End Date']
    )
    exprepdate.index = codes

    # 확장 벡터 매핑 연산 적용
    exprepdate['Expected FY'] = get_FQFY(exprepdate['Expected Period End Date'])
    exprepdate['Expected Next FY'] = get_next_FQFY(exprepdate['Expected Period End Date'])

    today_dt = dt.date.today()
    target_window_mask = (exprepdate['Expected Report Date'].dt.date >= today_dt) & \
                         (exprepdate['Expected Report Date'].dt.date <= today_dt + dt.timedelta(days=61))
    expreplist = exprepdate[target_window_mask].index.tolist()

    # 결과 전용 템플릿 프레임 생성
    exprep_df = pd.DataFrame(index=expreplist)
    exprep_df['Expected Report Date'] = exprepdate.loc[expreplist, 'Expected Report Date']
    exprep_df['Ticker'] = exprepdate.loc[expreplist, 'Ticker Symbol']
    exprep_df['Expected FY'] = exprepdate.loc[expreplist, 'Expected FY']
    exprep_df['Expected Next FY'] = exprepdate.loc[expreplist, 'Expected Next FY']
    exprep_df = exprep_df.reset_index().set_index(['index', 'Expected FY'])

    # 시계열 백데이터 정렬 후 전일자 데이터 생성
    equity_data = equity_data.sort_values(by=['Code', 'FY', 'EQBtw'])
    equity_data['EPS_EW_prev'] = equity_data['EPS_EW'].shift(-1)

    # 컴팩트 그룹화 집계 연산 처리
    equity_data_min = equity_data.groupby(['Code', 'FY'])[[
        'name', q_basis, 'GICS Sector', 'GICS Industry', 'model', 'EPS_Est', 'EPS_EW', 'EPS_EW_prev'
    ]].first()

    # FQ1 컬럼 바인딩
    exprep_df['Name'] = equity_data_min['name']
    exprep_df['GICS Sector'] = equity_data_min['GICS Sector']
    exprep_df['GICS Industry'] = equity_data_min['GICS Industry']
    exprep_df['FQ1 Model'] = equity_data_min['model']
    exprep_df['FQ1 Model EPS'] = equity_data_min['EPS_Est']
    exprep_df['FQ1 EW EPS'] = equity_data_min['EPS_EW']
    exprep_df['FQ1 EW Prev EPS'] = equity_data_min['EPS_EW_prev']

    # FQ2 컬럼 인덱스 스왑 바인딩
    exprep_df = exprep_df.reset_index().set_index(['index', 'Expected Next FY'])
    exprep_df['FQ2 Model'] = equity_data_min['model']
    exprep_df['FQ2 Model EPS'] = equity_data_min['EPS_Est']
    exprep_df['FQ2 EW EPS'] = equity_data_min['EPS_EW']
    exprep_df['FQ2 EW Prev EPS'] = equity_data_min['EPS_EW_prev']

    # 최종 결과 필드 정렬 체계 재배치
    exprep_df = exprep_df.reset_index().sort_values('Expected Report Date')
    target_cols = [
        'index', 'Ticker', 'Name', 'GICS Sector', 'GICS Industry', 'Expected FY', 'Expected Report Date',
        'FQ1 Model', 'FQ1 Model EPS', 'FQ1 EW EPS', 'FQ1 EW Prev EPS',
        'FQ2 Model', 'FQ2 Model EPS', 'FQ2 EW EPS', 'FQ2 EW Prev EPS'
    ]
    return exprep_df[target_cols]


def parse_args():
    args = argparse.ArgumentParser()
    args.add_argument('-c', '--country', default='kr', choices=['us','kr'], help='Country code (default: us)')
    args.add_argument('-q', '--quarter', default='2Q26AS', help='Current quarter (default: 2Q26AS)')
    args.add_argument('-b', '--basis', default='CQBtw', help='Basis (default: CQBtw)')
    args.add_argument('-l', '--loc', type=int, default=-1, help='Location of the recent quarter file (0: oldest, -1: latest)')
    return args.parse_args()

# ==============================================================================
# 4. 엔트리 포인트 오케스트레이터 (Main Control)
# ==============================================================================
def main():
    # 글로벌 제어 변수 선언 영역
    args = parse_args()
    country = args.country
    current_quarter = args.quarter
    q_basis = args.basis
    sector_len = 2 if country == 'us' else 3
    loc = args.loc  # 최근 분기 파일 선택 (0: 가장 오래된, -1: 가장 최신)

    # 환경 바인딩 초기화
    api_key = load_config()
    open_refinitiv_session(api_key)

    # 비즈니스 가공 파이프라인 구동
    final_result = process_earning_tracker(country, q_basis, current_quarter, loc)
    exprep_df = process_expected_equity(country, q_basis, sector_len)

    print('Writing to Excel File...')
    output_path = f'./result/earning_tracker_{country}/earning_tracker{loc}.xlsx'

    with pd.ExcelWriter(output_path) as writer:
        for i in range(4):
            sheet_name = f"Q{i + 1}"
            if i < len(final_result):
                final_result[i].to_excel(writer, sheet_name=sheet_name, index=False)

        sheet_name = "실적 발표 예정 종목"
        exprep_df.to_excel(writer, sheet_name=sheet_name, index=False)

    print("Pipeline Execution Completed Successfully.")


if __name__ == '__main__':
    main()