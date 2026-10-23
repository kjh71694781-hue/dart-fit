import os
import re
from datetime import datetime
import requests
import pandas as pd
import streamlit as st
from bs4 import BeautifulSoup
from dotenv import load_dotenv, find_dotenv
import OpenDartReader
from google import genai

# 1. 로컬 환경 변수 로드 (.env 자동 탐색)
load_dotenv(find_dotenv())

st.set_page_config(
    page_title="DART-Fit (다트핏) - AI 면접 브리핑 자동화",
    page_icon="🎯",
    layout="wide",
)

# --- 보안 API 키 로더 (Streamlit Cloud Secrets & 로컬 .env 모두 지원, 화면 노출 방지) ---
def get_secret_key(key_name: str) -> str:
    # 1순위: Streamlit Community Cloud의 st.secrets
    try:
        if key_name in st.secrets:
            return st.secrets[key_name]
    except Exception:
        pass
    # 2순위: 로컬 .env 환경변수
    return os.getenv(key_name, "")


# --- 핵심 데이터 수집 함수 ---
@st.cache_resource
def get_dart_client(api_key: str):
    return OpenDartReader(api_key)


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_financial_summary(api_key: str, corp_name: str, base_year: int = 2025):
    dart = get_dart_client(api_key)
    fin_summary = []
    target_accounts = ["매출액", "영업이익", "당기순이익", "자산총계", "부채총계", "자본총계"]

    for year in range(base_year - 2, base_year + 1):
        try:
            df = dart.finstate(corp_name, year)
            if df is not None and not df.empty:
                df_fs = df[df["fs_div"] == "CFS"]
                if df_fs.empty:
                    df_fs = df[df["fs_div"] == "OFS"]
                filtered = df_fs[df_fs["account_nm"].isin(target_accounts)]
                for _, row in filtered.iterrows():
                    fin_summary.append({
                        "연도": f"{year}년",
                        "계정과목": row["account_nm"],
                        "금액(원)": row["thstrm_amount"],
                    })
        except Exception:
            pass

    if not fin_summary:
        return pd.DataFrame(), "재무 데이터 조회 실패 (기업명을 정확히 입력했는지 확인하거나 기준 연도를 조정해 보세요)"

    df_fin = pd.DataFrame(fin_summary)
    df_pivot = (
        df_fin.pivot_table(index="계정과목", columns="연도", values="금액(원)", aggfunc="first")
        .reset_index()
    )
    return df_pivot, df_pivot.to_markdown(index=False)


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_recent_reports_and_contents(api_key: str, corp_name: str, base_year: int = 2025, max_chars: int = 2000):
    dart = get_dart_client(api_key)
    report_titles_md = "최근 정기공시 내역 없음"
    biz_contents_text = "사업보고서 본문 추출 내역 없음"
    df_reports_head = pd.DataFrame()

    try:
        recent_reports = dart.list(corp_name, start=f"{base_year - 1}-01-01", kind="A")
        if recent_reports is not None and not recent_reports.empty:
            df_reports_head = recent_reports[["rcept_dt", "report_nm", "rcept_no"]].head(5)
            report_titles_md = df_reports_head.to_markdown(index=False)

            latest_rcept_no = recent_reports.iloc[0]["rcept_no"]
            sub_docs = dart.sub_docs(latest_rcept_no)
            keywords = ["사업의 개요", "주요 제품", "연구개발", "신규사업", "사업의 내용"]
            target_docs = sub_docs[sub_docs["title"].str.contains("|".join(keywords), na=False)]
            if target_docs.empty:
                target_docs = sub_docs.head(4)

            extracted = []
            for _, doc in target_docs.head(4).iterrows():
                title = doc["title"]
                url = doc["url"]
                res = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=10)
                res.encoding = "utf-8"
                soup = BeautifulSoup(res.text, "lxml")
                text = re.sub(r"\s+", " ", soup.get_text(separator=" ", strip=True))
                extracted.append(f"### [{title}]\n{text[:max_chars]}...")
            if extracted:
                biz_contents_text = "\n\n".join(extracted)
    except Exception as e:
        biz_contents_text = f"공시 및 본문 조회 중 오류 발생: {e}"

    return df_reports_head, report_titles_md, biz_contents_text


def build_prompt(corp_name: str, job_role: str, fin_table_md: str, report_titles_md: str, biz_contents: str) -> str:
    return f"""당신은 대한민국 대기업 수석 면접관이자 기업 재무/전략 분석가입니다.
아래는 OpenDART API를 통해 추출한 [{corp_name}]의 실제 공시, 3개년 재무제표, 그리고 최신 사업보고서('사업의 내용') 원문 발췌 데이터입니다.
현재 [{job_role}] 직무에 지원하는 취업 준비생을 위해 실전 '면접 브리핑 리포트'를 작성해 주세요.

[1. 수집된 DART 재무 데이터 (3개년 추이)]
{fin_table_md}

[2. 최근 주요 정기공시 목록]
{report_titles_md}

[3. 최신 사업보고서 '사업의 내용' 및 '연구개발' 핵심 원문 발췌]
{biz_contents}

---
반드시 아래 4가지 섹션으로 나누어 구체적인 수치 및 사업보고서 내 고유명사(제품명, R&D 과제명 등)를 인용하며 작성하세요.
1. 📊 재무 흐름 3줄 요약 및 시사점 (매출/영업이익 증감률 및 재무 건전성 중심)
2. 🚨 면접관이 찌를 '아픈 손가락(Risk)'과 방어 논리 (실적 둔화, 부채비율, 신사업 경쟁 등)
3. 🎯 [{job_role}] 직무 지원자 전용 필살기 어필 포인트 (추출된 사업의 내용/R&D 현황과 지원 직무 연결)
4. 🔥 실전 압박 면접 예상 질문 3선 & 핵심 답변 가이드
"""


def generate_ai_report(gemini_api_key: str, prompt: str) -> str:
    client = genai.Client(api_key=gemini_api_key)
    interaction = client.interactions.create(
        model="gemini-3.8-flash",
        system_instruction="당신은 대한민국 대기업 수석 면접관이자 기업 재무/전략 분석가입니다. DART 공시 팩트에 기반해 날카롭고 실전적인 면접 가이드를 마크다운으로 작성합니다.",
        input=prompt,
    )
    return interaction.output_text


# --- 서버 기본 키 로드 (절대 UI 텍스트박스에 기본값으로 노출하지 않음) ---
server_dart_key = get_secret_key("OPENDART_API_KEY")
server_gemini_key = get_secret_key("GEMINI_API_KEY")

# --- UI 레이아웃 ---
st.title("🎯 DART-Fit (다트핏)")
st.subheader("300페이지 사업보고서를 1분 만에 직무 맞춤형 면접 브리핑 리포트로")

with st.sidebar:
    st.header("⚙️ 분석 옵션")
    base_year = st.number_input("기준 연도 (Base Year)", min_value=2020, max_value=2026, value=2025)

    st.markdown("---")
    st.subheader("🔑 API 연결 상태")
    if server_dart_key:
        st.success("✅ OpenDART 서버 키 연결됨")
    else:
        st.warning("⚠️ 서버에 OpenDART 키가 없습니다.")

    if server_gemini_key:
        st.success("✅ Gemini AI 서버 키 연결됨")
    else:
        st.info("ℹ️ Gemini 키 미설정 (프롬프트 생성 모드 작동)")

    with st.expander("개인 API Key 직접 입력 (선택사항)"):
        user_dart_key = st.text_input("개인 OpenDART API Key", type="password")
        user_gemini_key = st.text_input(
            "개인 Gemini API Key",
            type="password",
            help="서버 공용 한도 초과 시 본인 키를 입력해 사용할 수 있습니다.",
        )

active_dart_key = user_dart_key.strip() if user_dart_key else server_dart_key
active_gemini_key = user_gemini_key.strip() if user_gemini_key else server_gemini_key

col1, col2 = st.columns(2)
with col1:
    corp_name = st.text_input("🏢 분석할 기업명 (DART 정식 명칭)", value="SK텔레콤", placeholder="예: 삼성전자, SK하이닉스, 현대자동차")
with col2:
    job_role = st.text_input("💼 지원 직무", value="AI/빅데이터 경영 및 데이터 분석", placeholder="예: 반도체 공정엔지니어, B2B 마케팅, 재무회계")

if st.button("🚀 1분 면접 브리핑 리포트 생성하기", type="primary", use_container_width=True):
    if not active_dart_key:
        st.error("❌ OpenDART API Key가 설정되지 않았습니다. 관리자에게 문의하거나 좌측 사이드바에 개인 키를 입력해주세요.")
    else:
        with st.spinner(f"🔍 [{corp_name}]의 3개년 재무제표 및 최신 사업보고서 본문을 DART에서 수집 중입니다..."):
            df_fin, fin_md = fetch_financial_summary(active_dart_key, corp_name.strip(), base_year)
            df_reports, reports_md, biz_contents = fetch_recent_reports_and_contents(active_dart_key, corp_name.strip(), base_year)
            full_prompt = build_prompt(corp_name.strip(), job_role.strip(), fin_md, reports_md, biz_contents)

        st.success(f"✅ [{corp_name}] DART 공시 및 사업보고서 데이터 추출 완료!")

        tab1, tab2, tab3 = st.tabs([
            "🤖 AI 면접 브리핑 리포트",
            "📋 프롬프트 원문 (ChatGPT/Claude/Gemini 복붙용)",
            "📊 추출된 DART 재무/공시 원본",
        ])

        with tab1:
            if active_gemini_key:
                with st.spinner("🧠 Gemini AI가 공시 데이터를 분석하여 면접 브리핑 리포트를 작성 중입니다..."):
                    try:
                        ai_report = generate_ai_report(active_gemini_key, full_prompt)
                        st.markdown(ai_report)
                        st.download_button(
                            label="📥 마크다운 리포트(.md) 다운로드 (블로그/노션 발행용)",
                            data=ai_report,
                            file_name=f"DART_Fit_{corp_name}_{job_role}_리포트.md",
                            mime="text/markdown",
                        )
                    except Exception as e:
                        st.error(f"Gemini API 호출 중 오류가 발생했습니다: {e}")
            else:
                st.info(
                    "💡 현재 **프롬프트 생성 모드**입니다. 바로 옆 **[📋 프롬프트 원문]** 탭을 눌러 복사 버튼을 클릭한 뒤, 사용하시는 ChatGPT / Claude / Gemini 창에 붙여넣으시면 동일한 리포트가 출력됩니다!"
                )

        with tab2:
            st.markdown("아래 코드 블록 우측 상단의 **복사(Copy) 아이콘**을 클릭한 후 ChatGPT / Claude / Gemini에 붙여넣으세요!")
            st.code(full_prompt, language="markdown")
            st.download_button(
                label="📥 프롬프트 텍스트(.txt) 다운로드",
                data=full_prompt,
                file_name=f"interview_prompt_{corp_name}.txt",
                mime="text/plain",
            )

        with tab3:
            st.markdown("#### 1️⃣ 최근 3개년 핵심 재무추이")
            if not df_fin.empty:
                st.dataframe(df_fin, use_container_width=True)
            else:
                st.warning(fin_md)

            st.markdown("#### 2️⃣ 최근 정기공시 목록")
            if not df_reports.empty:
                st.dataframe(df_reports, use_container_width=True)
            else:
                st.warning(reports_md)

            st.markdown("#### 3️⃣ 최신 사업보고서 핵심 본문 발췌")
            with st.expander("원문 발췌 텍스트 펼쳐보기"):
                st.text(biz_contents)

# --- 실사용자 피드백 수집 섹션 (SK AIGEN 포트폴리오 정량 지표 확보용) ---
st.markdown("---")
with st.expander("💬 [10초 피드백] DART-Fit이 면접 준비 시간을 얼마나 줄여주었나요?"):
    with st.form("feedback_form"):
        f_col1, f_col2 = st.columns(2)
        with f_col1:
            saved_time = st.select_slider(
                "예상 단축 시간",
                options=["30분 미만", "약 1시간", "약 2시간", "3시간 이상 단축!"],
                value="3시간 이상 단축!",
            )
        with f_col2:
            rating = st.radio("리포트 만족도", ["⭐⭐⭐⭐⭐ (5점)", "⭐⭐⭐⭐ (4점)", "⭐⭐⭐ (3점)", "⭐⭐ 이하"], horizontal=True)
        comment = st.text_input("아쉬운 점이나 추가되었으면 하는 기능을 자유롭게 적어주세요!")
        submitted = st.form_submit_button("피드백 제출하기")
        if submitted:
            feedback_row = pd.DataFrame([{
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "corp_name": corp_name,
                "job_role": job_role,
                "saved_time": saved_time,
                "rating": rating,
                "comment": comment,
            }])
            feedback_file = "user_feedback.csv"
            feedback_row.to_csv(feedback_file, mode="a", header=not os.path.exists(feedback_file), index=False, encoding="utf-8-sig")
            st.success("소중한 피드백 감사합니다! 더 강력한 면접 분석 기능으로 보답하겠습니다. 🙌")
