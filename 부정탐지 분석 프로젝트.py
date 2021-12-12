###########################################
# 보험 사기자 예측, 사기자이면 1, 아니면 0 #
###########################################

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline

#데이터프레임을 화면에 출력할 때 보여야 하는 최대 열의 개수
pd.options.display.max_columns = 999

cust = pd.read_csv('csv/CUST_DATA.csv', encoding='utf-16')
claim = pd.read_csv('csv/CLAIM_DATA.csv', encoding='utf-16')

#########CUST_DATA#########

cust.shape
cust.info() 

cust.describe(include='all')

cust.hist(bins=50, figsize=(20, 15))
plt.show()

#결측치 분포 확인
import missingno
missingno.matrix(cust)
plt.show()

#############변수별 정보 확인#############

#일반인(N)과 사기자(Y)의 수 확인
cust['SIU_CUST_YN'].value_counts()

#FP(보험설계사) 경험 유(Y), 무(N) 확인
cust['FP_CAREER'].value_counts()

#고객의 거주 시/도별 거주자 수 확인
cust['CTPR'].value_counts()

#8개의 직업군으로 분류한 직업별 분포 확인
cust['OCCP_GRP_1'].value_counts()

#25개의 직업군으로 분류한 직업별 분포 확인
cust['OCCP_GRP_2'].value_counts()

#결혼 유(Y), 무(N) 확인
cust['WEDD_YN'].value_counts()

#8개의 직업군으로 분류한 배우자의 직업별 분포 확인
cust['MATE_OCCP_GRP_1'].value_counts()

#25개의 직업군으로 분류한 배우자의 직업별 분포 확인
cust['MATE_OCCP_GRP_2'].value_counts()

#########CLAIM_DATA#########
claim.describe(include='all')

claim.hist(bins=50, figsize=(30,20))

missingno.matrix(claim)

#############변수별 정보 확인#############

#금감원 유의 병원 대상 여부 정도
claim['HEED_HOSP_YN'].value_counts()

#유의 병원의 의사 라이센스 번호 체크 횟수
claim[claim['HEED_HOSP_YN']=='Y']['CHME_LICE_NO'].value_counts()

#보험 청구를 많이 할 수록 사기자일 확률이 높을 것이란 가정하에 보험 청구횟수 확인
claim['CUST_ID'].value_counts()[claim['CUST_ID'].value_counts() > 50]

cust[cust['CUST_ID']==15109]


###########CUST_DATA 전처리###########

#나이 관련 변수를 연령대로 변환
cust['AGE'] = cust['AGE'].map(lambda x : int(x//10))
cust.head()

cust['LTBN_CHLD_AGE'] = cust['LTBN_CHLD_AGE'].map(lambda x : x//10)
cust.head()

#OCCP_GRP_2 변수 삭제
cust.drop('OCCP_GRP_2', axis=1, inplace=True)
cust.head()

#OCCP_GRP_1 변수에서 코드번호 삭제
set(cust.OCCP_GRP_1)
cust['OCCP_GRP_1'] = cust['OCCP_GRP_1'].map(lambda x : str(x)[2:])
cust.head()

#MATE_OCCP_GRP_2 변수 삭제
cust.drop('MATE_OCCP_GRP_2', axis=1, inplace=True)

#MATE_OCCP_GRP_1 변수에서 코드번호 삭제
set(cust.MATE_OCCP_GRP_1)
cust['MATE_OCCP_GRP_1'] = cust['MATE_OCCP_GRP_1'].map(lambda x : str(x)[2:])
cust.head()

#DATE 관련 변수 삭제
cust.drop(['CUST_RGST', 'MAX_PAYM_YM'], axis=1, inplace=True)
cust.head()

#null 값 확인
missingno.bar(cust)

cust.isnull().sum()

#null 값이 많은 MINCRDT, MAXCRDT 열 삭제
cust.drop(['MINCRDT', 'MAXCRDT'], axis=1, inplace=True)
cust.head()

missingno.bar(cust)

#개인소득 결측치 처리
#고객의 연령/직업/보험료 수준 등을 통한 고객의 개인소득(CUST_INCM) 추정
cust[cust.CUST_INCM.isnull()].head()

occp_age_cust = cust.pivot_table(index=['AGE', 'OCCP_GRP_1'],
                                 values='CUST_INCM', aggfunc='mean')
occp_age_cust

def fill_income(row):
    try:
        avg_income = occp_age_cust['CUST_INCM'][row.AGE][row.OCCP_GRP_1]
    except:
        avg_income = 0
    return avg_income

cust['CUST_INCM_NEW'] = None
cust['CUST_INCM_NEW'][cust.CUST_INCM.isnull()] = cust[cust.CUST_INCM.isnull()].apply(fill_income, axis=1)
missingno.bar(cust)

cust['CUST_INCM_NEW'][cust.CUST_INCM.isnull()] = cust[cust.CUST_INCM.isnull()].apply(fill_income, axis=1)
del cust['CUST_INCM_NEW']
cust.head()

missingno.bar(cust)

#가구 소득 결측치 처리
avg_income_by_job = cust.pivot_table(index=['OCCP_GRP_1'],
                                     values=['JPBASE_HSHD_INCM'],
                                     aggfunc='mean')
avg_income_by_job

def fill_jbbase_incm(row):
    try:
        avg_jpbase = jpbase_df['JPBASE_HSHD_INCM'][row.OCCP_GRP_1]
    except:
        avg_jpbase = 0
    return avg_jpbase

cust['JPBASE_NEW'] = None
cust['JPBASE_NEW'][cust.JPBASE_HSHD_INCM.isnull()] = cust[cust['JPBASE_HSHD_INCM'].isnull()].apply(fill_jbbase_incm, axis=1)

missingno.bar(cust)

cust['JPBASE_HSHD_INCM'][cust.JPBASE_HSHD_INCM.isnull()] = cust[cust['JPBASE_HSHD_INCM'].isnull()].apply(fill_jbbase_incm, axis=1)
del cust['JPBASE_NEW']

missingno.bar(cust)

##########기타 변수 결측치 처리##########
cust['RESI_TYPE_CODE'].value_counts()

#RESI_TYPE_CODE, CTPR : 결측치는 최빈값으로
cust['RESI_TYPE_CODE'].fillna(20, inplace=True)
cust['CTPR'].fillna('경기', inplace=True)

#WEDD_YN : 결측치는 N으로
cust['WEDD_YN'].fillna('N', inplace=True)

#LTBN_CHLD_AGE, CHLD_CNT, TOTALPREM, MAX_PRM : 결측치는 0으로
cust['LTBN_CHLD_AGE'].fillna(0, inplace=True)
cust['CHLD_CNT'].fillna(0, inplace=True)
cust['TOTALPREM'].fillna(0, inplace=True)
cust['MAX_PRM'].fillna(0, inplace=True)

missingno.bar(cust)

##########중간 데이터 저장##########
cust.to_csv('CUST_DATA_결측치처리.csv', index=False, encoding='utf-8-sig')

#분석에 유의미하지 않다고 판단하는 열(FP_CAREER) DROP
cust.drop('FP_CAREER', axis=1, inplace=True)
cust.head()

#카테고리 변수 원-핫 인코딩
category = ['SEX', 'RESI_TYPE_CODE', 'CTPR', 'OCCP_GRP_1', 'WEDD_YN', 'MATE_OCCP_GRP_1']
dummy = pd.get_dummies(cust, columns=category)
dummy.head()

#원-핫 인코딩 된 데이터 저장
dummy.to_csv('CUST_DATA_더미코딩후.csv', index=False, encoding='utf-8-sig')
del dummy

#CLAIM_DATA로부터 파생변수 생성
cust = pd.read_csv('CUST_DATA_더미코딩후.csv', encoding='utf-8-sig')
claim = pd.read_csv('csv/CLAIM_DATA.csv', encoding='utf-16')

claim.head()
missingno.bar(claim)

missingno.matrix(claim)

#결측치가 많은 변수 및 날짜 관련 변수 삭제

#NULL 값 많은 변수 삭제
claim.drop(['SELF_CHAM', 'NON_PAY', 'TAMT_SFCA', 'PATT_CHRG_TOTA',
            'DSCT_AMT', 'COUNT_TRMT_ITEM', 'DCAF_CMPS_XCPA'],
            axis=1, inplace=True)

#DATE 변수 삭제
claim.drop(['HOSP_OTPA_STDT', 'HOSP_OTPA_ENDT'], axis=1, inplace=True)

missingno.bar(claim)

#HOSP_DAYS : 고객별 평균 입원 일수
claim_df = claim.pivot_table(index='CUST_ID',
                             values='VLID_HOSP_OTDA',
                             aggfunc='mean')

claim_df.reset_index(inplace=True)
cust = pd.merge(cust, claim_df, how='left', on='CUST_ID')
cust.rename(columns={'VLID_HOSP_OTDA':'HOSP_DAYS'}, inplace=True)
cust.loc[:, ['CUST_ID', 'HOSP_DAYS']].head(10)

#HEED_HOSP : 고객별 유의병원 출입여부
id_heed = claim[claim['HEED_HOSP_YN']=='Y']['CUST_ID'].unique()
id_heed

cust['HEED_HOSP'] = np.nan
cust['HEED_HOSP'][cust['CUST_ID'].isin(id_heed)] = 1
cust['HEED_HOSP'].fillna(0, inplace=True)
cust['HEED_HOSP'].value_counts()

#CLAIM_COUNT : 고객별 청구횟수
청구횟수 = claim.pivot_table(index='CUST_ID',
                             values='POLY_NO',
                             aggfunc='count').reset_index()
청구횟수.columns = ['CUST_ID', 'CLAIM_COUNT']
cust = pd.merge(cust, 청구횟수, on='CUST_ID', how='left')

#DOC_SIU_RATIO : 의사별 사기 비율

#의사별 사기수/전체청구수
data = pd.merge(claim, cust[['CUST_ID', 'SIU_CUST_YN']], how='left', on='CUST_ID')
의사별전체 = data.pivot_table(index=['CHME_LICE_NO'],
                              values='POLY_NO',
                              aggfunc='count').reset_index()
의사별_YN = data.pivot_table(index=['CHME_LICE_NO', 'SIU_CUST_YN'],
                              values='POLY_NO',
                              aggfunc='count').reset_index()
의사별_YN = 의사별_YN[의사별_YN['SIU_CUST_YN']=='Y']
의사별전체 = pd.merge(의사별전체, 의사별_YN, how='left', on='CHME_LICE_NO')
의사별전체['DOC_SIU_RATIO'] = 의사별전체['POLY_NO_y'] / 의사별전체['POLY_NO_x']
의사별전체.fillna(0, inplace=True)
의사별사기비율 = 의사별전체[['CHME_LICE_NO', 'DOC_SIU_RATIO']]
의사별사기비율.head()
