import os
import re
import io
import html
import zipfile
import xml.etree.ElementTree as ET
from datetime import datetime
import requests
import pandas as pd
import streamlit as st

# 로컬 .env 파일이 있으면 선택적으로 로드 (외부 패키지 python-dotenv 없어도 작동!)
try:
    from dotenv import load_dotenv, find_dotenv
    load_dotenv(find_dotenv())
except ModuleNotFoundError:
    if os.path.exists(".env"):
        with open(".env", encoding="utf-8") as f:
            for line in f:
                if "=" in line and not line.strip().startswith("#"):
                    k, v = line.strip().split("=", 1)
                    os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))

st.set_page_config(
    page_title="DART-Fit (다트핏) - AI 면접 브리핑 자동화",
    page_icon="🎯",
    layout="wide",
)


# --- 1. 외부 패키지 의존성 없는 순수 Python 마크다운 표 변환기 ---
def df_to_md(df: pd.DataFrame) -> str:
    if df is None or df.empty:
        return "데이터 없음"
    cols = [str(c) for c in df.columns]
    header = "| " + " | ".join(cols) + " |"
    sep = "| " + " | ".join(["---"] * len(cols)) + " |"
    rows = []
    for _, row in df.iterrows():
        rows.append("| " + " | ".join(str(val) if pd.notna(val) else "-" for val in row) + " |")
    return "\n".join([header, sep] + rows)


# --- 2. 외부 패키지 의존성 없는 순수 Python HTML 본문 텍스트 정제기 ---
def clean_html_text(raw_html: str) -> str:
    text = re.sub(r"<(script|style)[^>]*>.*?</\1>", " ", raw_html, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"<[^>]+>", " ", text)
    text = html.unescape(text)
    return re.sub(r"\s+", " ", text).strip()


# --- 3. 내장형 OpenDART API 클라이언트 (OpenDartReader 설치 없이도 100% 작동!) ---
class BuiltinDartClient:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.corp_codes = self._load_corp_codes()

    def _load_corp_codes(self) -> pd.DataFrame:
        url = "https://opendart.fss.or.kr/api/corpCode.xml"
        r = requests.get(url, params={"crtfc_key": self.api_key}, timeout=15)
        zf = zipfile.ZipFile(io.BytesIO(r.content))
        xml_data = zf.read("CORPCODE.xml")
        tree = ET.XML(xml_data)
        records = []
        for child in tree.findall("list"):
            rec = {sub.tag: sub.text for sub in child}
            records.append(rec)
        return pd.DataFrame(records)

    def find_corp_code(self, corp: str) -> str:
        if corp.isdigit() and len(corp) == 8:
            return corp
        df = self.corp_codes
        listed = df[df["stock_code"].notna() & (df["stock_code"].str.strip() != "")]
        matched = listed[listed["corp_name"] == corp]
        if matched.empty:
            matched = df[df["corp_name"] == corp]
        if matched.empty:
            matched = listed[listed["corp_name"].str.contains(re.escape(corp), na=False)]
        if matched.empty:
            return ""
        return matched.iloc[0]["corp_code"]

    def finstate(self, corp: str, bsns_year: int, reprt_code: str = "11011") -> pd.DataFrame:
        corp_code = self.find_corp_code(corp)
        if not corp_code:
            return pd.DataFrame()
        url = "https://opendart.fss.or.kr/api/fnlttSinglAcnt.json"
        params = {
            "crtfc_key": self.api_key,
            "corp_code": corp_code,
            "bsns_year": str(bsns_year),
            "reprt_code": reprt_code,
        }
        r = requests.get(url, params=params, timeout=10).json()
        if "list" not in r:
            return pd.DataFrame()
        return pd.DataFrame(r["list"])

    def list_reports(self, corp: str, start: str = "2024-01-01", kind: str = "A") -> pd.DataFrame:
        corp_code = self.find_corp_code(corp)
        if not corp_code:
            return pd.DataFrame()
        url = "https://opendart.fss.or.kr/api/list.json"
        bgn_de = start.replace("-", "")
        params = {
            "crtfc_key": self.api_key,
            "corp_code": corp_code,
            "bgn_de": bgn_de,
            "pblntf_ty": kind,
            "page_no": 1,
            "page_count": 30,
        }
        r = requests.get(url, params=params, timeout=10).json()
        if "list" not in r:
            return pd.DataFrame()
        return pd.DataFrame(r["list"])

    def sub_docs(self, rcp_no: str) -> pd.DataFrame:
        url = f"http://dart.fss.or.kr/dsaf001/main.do?rcpNo={rcp_no}"
        r = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=10)
        multi_page_re = (
            r"\s+node[12]\['text'\][ =]+\"(.*?)\"\;"
            r"\s+node[12]\['id'\][ =]+\"(\d+)\";"
            r"\s+node[12]\['rcpNo'\][ =]+\"(\d+)\";"
            r"\s+node[12]\['dcmNo'\][ =]+\"(\d+)\";"
            r"\s+node[12]\['eleId'\][ =]+\"(\d+)\";"
            r"\s+node[12]\['offset'\][ =]+\"(\d+)\";"
            r"\s+node[12]\['length'\][ =]+\"(\d+)\";"
            r"\s+node[12]\['dtd'\][ =]+\"(.*?)\";"
            r"\s+node[12]\['tocNo'\][ =]+\"(\d+)\";"
        )
        matches = re.findall(multi_page_re, r.text)
        rows = []
        for m in matches:
            doc_title = m[0]
            params = f"rcpNo={m[2]}&dcmNo={m[3]}&eleId={m[4]}&offset={m[5]}&length={m[6]}&dtd={m[7]}"
            doc_url = f"http://dart.fss.or.kr/report/viewer.do?{params}"
            rows.append({"title": doc_title, "url": doc_url})
        return pd.DataFrame(rows)


# --- 보안 API 키 로더 ---
def get_secret_key(key_name: str) -> str:
    try:
        if key_name in st.secrets:
            return str(st.secrets[key_name]).strip()
    except Exception:
        pass
    return os.getenv(key_name, "").strip()


@st.cache_resource(show_spinner=False)
def get_dart_client(api_key: str) -> BuiltinDartClient:
    return BuiltinDartClient(api_key)


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
    return df_pivot, df_to_md(df_pivot)


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_recent_reports_and_contents(api_key: str, corp_name: str, base_year: int = 2025, max_chars: int = 2000):
    dart = get_dart_client(api_key)
    report_titles_md = "최근 정기공시 내역 없음"
    biz_contents_text = "사업보고서 본문 추출 내역 없음"
    df_reports_head = pd.DataFrame()

    try:
        recent_reports = dart.list_reports(corp_name, start=f"{base_year - 1}-01-01", kind="A")
        if recent_reports is not None and not recent_reports.empty:
            df_reports_head = recent_reports[["rcept_dt", "report_nm", "rcept_no"]].head(5)
            report_titles_md = df_to_md(df_reports_head)

            latest_rcept_no = str(recent_reports.iloc[0]["rcept_no"])
            sub_docs = dart.sub_docs(latest_rcept_no)
            if not sub_docs.empty:
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
                    text = clean_html_text(res.text)
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
    sys_inst = "당신은 대한민국 대기업 수석 면접관이자 기업 재무/전략 분석가입니다. DART 공시 팩트에 기반해 날카롭고 실전적인 면접 가이드를 마크다운으로 작성합니다."
    try:
        from google import genai
        client = genai.Client(api_key=gemini_api_key)
        interaction = client.interactions.create(
            model="gemini-3.8-flash",
            system_instruction=sys_inst,
            input=prompt,
        )
        return interaction.output_text
    except ModuleNotFoundError:
        url = f"https://generativelanguage.googleapis.com/v1/interactions?key={gemini_api_key}"
        payload = {
            "model": "gemini-3.8-flash",
            "system_instruction": sys_inst,
            "input": prompt,
        }
        r = requests.post(url, json=payload, timeout=60)
        r.raise_for_status()
        data = r.json()
        return data.get("output_text", "")


# --- 서버 기본 키 로드 ---
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
    job_role = st.text_input("💼 지원 직무", value="AI/
