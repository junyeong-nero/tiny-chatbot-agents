"""ToS 기반 Golden Set 생성 스크립트.

약관 조항 참조가 필요한 평가 데이터셋을 생성합니다.
각 질문에 expected_sources를 포함하여 검색 정확도 평가를 가능하게 합니다.
"""

import json
from pathlib import Path


def create_tos_golden_set() -> list[dict]:
    """약관 기반 Golden Set 생성."""

    tos_golden_set = [
        # === CMS출금이체약관 관련 ===
        {
            "id": "tos_001",
            "question": "CMS 출금이체에서 지정출금일이 휴일인 경우 어떻게 처리되나요?",
            "expected_answer": "지정출금일이 휴일인 경우 다음 영업일에 출금 대체 납부됩니다. [참조: CMS출금이체약관 제3조]",
            "category": "약관_CMS",
            "difficulty": "medium",
            "expected_sources": ["CMS출금이체약관", "제3조(출금)"]
        },
        {
            "id": "tos_002",
            "question": "출금이체 지정계좌 잔액이 청구금액보다 부족하면 어떻게 되나요?",
            "expected_answer": "인출가능금액이 청구금액보다 부족하여 대체납부가 불가능한 경우의 손해는 납부자의 책임입니다. [참조: CMS출금이체약관 제4조]",
            "category": "약관_CMS",
            "difficulty": "medium",
            "expected_sources": ["CMS출금이체약관", "제4조(과실책임)"]
        },
        {
            "id": "tos_003",
            "question": "CMS 출금이체 해지는 어떻게 하나요?",
            "expected_answer": "서면, 전화 녹취, ARS 또는 전자금융거래법령에 따른 방법으로 해지 의사표시를 하면 됩니다. 1년 이상 출금이체 이용 실적이 없는 경우 회사가 임의로 해지할 수 있습니다. [참조: CMS출금이체약관 제9조]",
            "category": "약관_CMS",
            "difficulty": "medium",
            "expected_sources": ["CMS출금이체약관", "제9조(출금이체 해지)"]
        },
        {
            "id": "tos_004",
            "question": "부분출금이 뭔가요?",
            "expected_answer": "부분출금방식으로 승인된 이용기관의 경우, 청구금액에 비해 인출가능금액이 부족해도 잔액 전액을 출금할 수 있습니다. 단, 수표 등이 부도처리되면 전액 출금취소됩니다. [참조: CMS출금이체약관 제5조]",
            "category": "약관_CMS",
            "difficulty": "hard",
            "expected_sources": ["CMS출금이체약관", "제5조(부분출금)"]
        },

        # === 개인정보처리방침 관련 ===
        {
            "id": "tos_005",
            "question": "한국투자증권이 수집하는 개인정보는 얼마나 보관되나요?",
            "expected_answer": "개인정보의 처리 및 보유기간은 법령에서 정한 기간 또는 정보주체로부터 동의받은 기간 동안 보유됩니다. 구체적인 보유기간은 개인정보처리방침 제2조에서 확인할 수 있습니다. [참조: 개인정보처리방침 제2조]",
            "category": "약관_개인정보",
            "difficulty": "medium",
            "expected_sources": ["개인정보처리방침", "제2조 개인정보의 처리 및 보유기간"]
        },
        {
            "id": "tos_006",
            "question": "만 14세 미만 아동도 계좌를 개설할 수 있나요?",
            "expected_answer": "만 14세 미만 아동의 개인정보 처리는 별도 규정이 적용됩니다. 법정대리인의 동의가 필요하며, 자세한 내용은 개인정보처리방침 제4조를 참조하세요. [참조: 개인정보처리방침 제4조]",
            "category": "약관_개인정보",
            "difficulty": "hard",
            "expected_sources": ["개인정보처리방침", "제4조 만14세미만 아동의 개인정보 처리"]
        },
        {
            "id": "tos_007",
            "question": "내 개인정보가 제3자에게 제공되나요?",
            "expected_answer": "개인정보의 제3자 제공에 관한 사항은 개인정보처리방침 제5조에 명시되어 있습니다. 정보주체의 동의를 받은 경우 또는 법률에 특별한 규정이 있는 경우에 한해 제공됩니다. [참조: 개인정보처리방침 제5조]",
            "category": "약관_개인정보",
            "difficulty": "medium",
            "expected_sources": ["개인정보처리방침", "제5조 개인정보의 제3자 제공에 관한 사항"]
        },
        {
            "id": "tos_008",
            "question": "개인정보 파기는 어떻게 이루어지나요?",
            "expected_answer": "보유기간이 경과하거나 처리목적이 달성된 개인정보는 지체없이 파기됩니다. 파기 절차 및 방법은 개인정보처리방침 제10조에서 확인할 수 있습니다. [참조: 개인정보처리방침 제10조]",
            "category": "약관_개인정보",
            "difficulty": "medium",
            "expected_sources": ["개인정보처리방침", "제10조 개인정보의 파기 절차 및 방법에 관한 사항"]
        },

        # === ELW 관련 ===
        {
            "id": "tos_009",
            "question": "ELW 투자 시 발행자의 신용위험은 어떻게 되나요?",
            "expected_answer": "ELW는 담보나 보증 없이 발행자의 신용으로 발행되므로 만기 시 발행자의 재무 상태에 따라 결제가 불이행될 위험이 있습니다. 거래소는 발행자의 결제이행의무를 보증하지 않습니다. [참조: ELW 투자시 유의사항]",
            "category": "약관_파생상품",
            "difficulty": "hard",
            "expected_sources": ["ELW 투자시 유의사항"]
        },
        {
            "id": "tos_010",
            "question": "ELW는 시간이 지나면 가치가 어떻게 되나요?",
            "expected_answer": "ELW는 시간경과에 따른 가치감소 상품입니다. 만기일에 가까워질수록 가치가 감소하여 결국 0 또는 행사가치에 접근합니다. 기초자산 가격이 변동하지 않아도 시간이 경과하면 가격이 하락합니다. [참조: ELW 투자시 유의사항]",
            "category": "약관_파생상품",
            "difficulty": "hard",
            "expected_sources": ["ELW 투자시 유의사항"]
        },

        # === EUREX 연계선물 관련 ===
        {
            "id": "tos_011",
            "question": "EUREX 연계선물 거래 시간은 어떻게 되나요?",
            "expected_answer": "거래시간은 오후 6시부터 익일 오전 5시까지입니다. 유럽의 일광시간절약제가 적용되는 기간에는 오전 4시까지입니다. [참조: EUREX 연계선물 위험고지서 및 거래 설명서]",
            "category": "약관_파생상품",
            "difficulty": "medium",
            "expected_sources": ["EUREX 연계선물 위험고지서 및 거래 설명서"]
        },
        {
            "id": "tos_012",
            "question": "EUREX 연계선물의 위탁증거금이 부족하면 어떻게 되나요?",
            "expected_answer": "위탁증거금을 정해진 기한까지 예탁하지 않은 경우 회사는 고객의 미결제약정이나 대용증권 등의 일부를 처분하여 위탁증거금에 충당하며, 이로 인한 손실은 고객이 부담합니다. [참조: EUREX 연계선물 위험고지서]",
            "category": "약관_파생상품",
            "difficulty": "hard",
            "expected_sources": ["EUREX 연계선물 위험고지서 및 거래 설명서"]
        },
        {
            "id": "tos_013",
            "question": "일중매매거래(Day-trading)는 어떤 투자자에게 적합하지 않나요?",
            "expected_answer": "일중매매거래는 제한된 자금, 투자경험의 부족, 낮은 위험선호도의 고객에게는 적절하지 않습니다. 직장인, 자영업자, 학생 등 생업이나 학업에 종사하는 고객에게도 적합하지 않습니다. [참조: 일중매매거래 위험고지]",
            "category": "약관_파생상품",
            "difficulty": "medium",
            "expected_sources": ["EUREX 연계선물 위험고지서 및 거래 설명서", "일중매매거래 위험고지"]
        },

        # === ISA 관련 ===
        {
            "id": "tos_014",
            "question": "ISA 중개형 계좌의 특징은 무엇인가요?",
            "expected_answer": "ISA 중개형(개인종합자산관리계좌)은 다양한 금융상품을 하나의 계좌에서 거래할 수 있으며, 손익통산과 세제혜택이 있습니다. 연간 납입한도 2천만원, 총 납입한도 1억원입니다. [참조: ISA 중개형 약관]",
            "category": "약관_ISA",
            "difficulty": "medium",
            "expected_sources": ["ISA 중개형(개인종합자산관리계좌) 약관", "ISA 중개형(개인종합자산관리계좌) 상품 설명서"]
        },
        {
            "id": "tos_015",
            "question": "ISA 신탁형과 중개형의 차이점은 무엇인가요?",
            "expected_answer": "ISA 신탁형은 회사가 고객의 재산을 관리하는 신탁계약 방식이고, ISA 중개형은 고객이 직접 투자상품을 선택하는 중개 방식입니다. 자세한 차이는 각 상품 설명서를 참조하세요. [참조: ISA 신탁형/중개형 설명서]",
            "category": "약관_ISA",
            "difficulty": "hard",
            "expected_sources": ["ISA 신탁형(종합자산관리신탁계약) 설명서", "ISA 중개형(개인종합자산관리계좌) 상품 설명서"]
        },

        # === CMA 관련 ===
        {
            "id": "tos_016",
            "question": "RP형 CMA와 MMF형 CMA의 차이점은 무엇인가요?",
            "expected_answer": "RP형 CMA는 환매조건부채권에 투자하여 예금자보호 대상이고, MMF형 CMA는 MMF(머니마켓펀드)에 투자하여 예금자보호 대상이 아닙니다. 수익률과 안정성이 다릅니다. [참조: RP형 CMA 설명서, MMF형 CMA설명서]",
            "category": "약관_CMA",
            "difficulty": "hard",
            "expected_sources": ["RP형 CMA 상품설명서", "MMF형 CMA설명서"]
        },
        {
            "id": "tos_017",
            "question": "CMA 계좌에서 수익이 발생하는 원리는 무엇인가요?",
            "expected_answer": "CMA는 예탁된 자금을 RP, MMF 등에 투자하여 수익을 발생시킵니다. 투자 대상에 따라 RP형, MMF형 등으로 구분되며, 각각의 투자구조와 수익률이 다릅니다. [참조: CMA 상품설명서]",
            "category": "약관_CMA",
            "difficulty": "medium",
            "expected_sources": ["RP 및 RP형 CMA 설명서", "MMF형 CMA설명서"]
        },

        # === 약관 변경 관련 ===
        {
            "id": "tos_018",
            "question": "약관이 변경되면 어떻게 알 수 있나요?",
            "expected_answer": "회사는 약관 변경 시 시행일 1개월 전에 영업점, 인터넷 홈페이지, 전자통신매체를 통해 게시합니다. 고객에게 불리한 변경인 경우 서면 등으로 사전에 통지합니다. [참조: 약관 변경 조항]",
            "category": "약관_일반",
            "difficulty": "medium",
            "expected_sources": ["CMS출금이체약관", "제14조(약관의 변경)"]
        },
        {
            "id": "tos_019",
            "question": "약관 변경에 동의하지 않으면 어떻게 되나요?",
            "expected_answer": "약관 변경에 동의하지 않는 경우 계약을 해지할 수 있습니다. 통지를 받은 날로부터 시행일 전 영업일까지 해지 의사표시를 하지 않으면 변경에 동의한 것으로 봅니다. [참조: 약관 변경 조항]",
            "category": "약관_일반",
            "difficulty": "hard",
            "expected_sources": ["CMS출금이체약관", "제14조(약관의 변경)"]
        },

        # === K-OTC 관련 ===
        {
            "id": "tos_020",
            "question": "K-OTC 시장에서 거래할 수 있는 종목은 무엇인가요?",
            "expected_answer": "K-OTC는 비상장주식을 거래할 수 있는 장외시장입니다. 금융위원회가 지정한 K-OTC 시장에서 호가중개 방식으로 비상장주식을 거래할 수 있습니다. [참조: K-OTC 거래설명서]",
            "category": "약관_K-OTC",
            "difficulty": "medium",
            "expected_sources": ["K-OTC 거래설명서"]
        },
        {
            "id": "tos_021",
            "question": "K-OTC 거래의 결제는 어떻게 이루어지나요?",
            "expected_answer": "K-OTC 거래는 매매일로부터 T+2일에 결제됩니다. 거래대금과 주식의 수도는 증권예탁원을 통해 이루어집니다. [참조: K-OTC 거래설명서]",
            "category": "약관_K-OTC",
            "difficulty": "medium",
            "expected_sources": ["K-OTC 거래설명서"]
        },

        # === 금융소비자 권리 관련 ===
        {
            "id": "tos_022",
            "question": "자료열람요구권이 무엇인가요?",
            "expected_answer": "분쟁조정 또는 소송의 수행 등 권리 구제를 위한 목적으로 금융소비자보호법 제28조에 따라 회사가 기록 및 유지, 관리하는 자료의 열람을 요구할 수 있는 권리입니다. 열람 요구일로부터 6영업일 이내 열람이 가능합니다. [참조: 금융소비자의 권리]",
            "category": "약관_소비자보호",
            "difficulty": "hard",
            "expected_sources": ["금융소비자의 권리", "자료열람요구권"]
        },
        {
            "id": "tos_023",
            "question": "위법계약의 해지는 언제까지 요구할 수 있나요?",
            "expected_answer": "회사의 위법사실을 안 날로부터 1년이 되는 날 또는 계약체결일로부터 5년이 되는 날 중 먼저 도달하는 날까지 서면 등의 방법으로 계약의 해지 요구가 가능합니다. [참조: 위법계약의 해지]",
            "category": "약관_소비자보호",
            "difficulty": "hard",
            "expected_sources": ["금융소비자의 권리", "위법계약의 해지"]
        },
        {
            "id": "tos_024",
            "question": "분쟁이 발생하면 어디에 조정을 신청할 수 있나요?",
            "expected_answer": "금융투자협회(02-2003-9426), 한국거래소(1577-2172), 금융감독원(1332)에 분쟁조정을 신청할 수 있습니다. 금융감독원 분쟁조정 수용 시 재판상 화해 효력이 발생합니다. [참조: 민원처리, 분쟁조정 절차]",
            "category": "약관_소비자보호",
            "difficulty": "medium",
            "expected_sources": ["민원처리, 분쟁조정 절차에 관한 사항"]
        },

        # === 예금자보호 관련 ===
        {
            "id": "tos_025",
            "question": "EUREX 연계선물은 예금자보호를 받을 수 있나요?",
            "expected_answer": "EUREX 연계선물은 예금자보호법에 따라 보호되지 않습니다. 파생상품 거래는 원금 손실 위험이 있으며, 예금자보호 대상이 아닙니다. [참조: EUREX 연계선물 위험고지서]",
            "category": "약관_예금자보호",
            "difficulty": "medium",
            "expected_sources": ["EUREX 연계선물 위험고지서 및 거래 설명서"]
        },

        # === 투자위험등급 관련 ===
        {
            "id": "tos_026",
            "question": "장내파생상품의 투자위험등급은 어떻게 되나요?",
            "expected_answer": "장내파생상품은 투자위험등급 1등급(매우높은위험)으로 분류됩니다. 투자원금을 초과하는 손실이 발생할 수 있으며, 공격투자형 투자자에게 적합한 상품입니다. [참조: 장내파생상품 위험고지서]",
            "category": "약관_파생상품",
            "difficulty": "medium",
            "expected_sources": ["EUREX 연계선물 위험고지서 및 거래 설명서"]
        },

        # === 복합 질문 ===
        {
            "id": "tos_027",
            "question": "CMS 출금이체 약관에서 영업일이란 무엇인가요?",
            "expected_answer": "영업일은 대통령령에 의한 관공서의 공휴일, 토요일, 근로자의 날을 제외한 날입니다. [참조: CMS출금이체약관 제2조]",
            "category": "약관_CMS",
            "difficulty": "easy",
            "expected_sources": ["CMS출금이체약관", "제2조(용어의 정의)"]
        },
        {
            "id": "tos_028",
            "question": "자동이체 통합관리시스템(Payinfo)에서 할 수 있는 것은 무엇인가요?",
            "expected_answer": "자동이체 통합관리시스템(www.payinfo.or.kr)에서 여러 금융회사에 등록된 지로/CMS출금이체 등록정보를 일괄 조회하고 해지 및 변경 신청이 가능합니다. [참조: CMS출금이체약관 제2조, 제12조]",
            "category": "약관_CMS",
            "difficulty": "medium",
            "expected_sources": ["CMS출금이체약관", "제2조(용어의 정의)", "제12조(자동이체 통합관리시스템)"]
        },
        {
            "id": "tos_029",
            "question": "회사의 민원 담당부서 연락처는 어디인가요?",
            "expected_answer": "본사 민원담당부서(소비자지원부)는 전화 02-3276-4334, 이메일 minwon@koreainvestment.com으로 연락할 수 있습니다. FAX는 02-3276-4332입니다. [참조: 민원처리 절차]",
            "category": "약관_민원",
            "difficulty": "easy",
            "expected_sources": ["민원처리, 분쟁조정 절차에 관한 사항"]
        },
        {
            "id": "tos_030",
            "question": "청약철회권은 EUREX 연계선물에 적용되나요?",
            "expected_answer": "EUREX 연계선물은 청약철회권을 행사할 수 있는 대상 상품이 아닙니다. 장내파생상품 거래는 청약철회 대상에서 제외됩니다. [참조: 금융소비자의 권리 - 청약철회권]",
            "category": "약관_소비자보호",
            "difficulty": "hard",
            "expected_sources": ["EUREX 연계선물 위험고지서 및 거래 설명서", "금융소비자의 권리"]
        },

        # === 추가 약관 질문 ===
        {
            "id": "tos_031",
            "question": "NICE지키미 서비스는 무엇인가요?",
            "expected_answer": "NICE지키미는 개인신용정보 조회 및 관리 서비스입니다. 회원 이용약관에 따라 본인의 신용정보를 조회하고 관리할 수 있습니다. [참조: NICE지키미 회원 이용약관]",
            "category": "약관_신용정보",
            "difficulty": "medium",
            "expected_sources": ["NICE지키미 회원 이용약관"]
        },
        {
            "id": "tos_032",
            "question": "MMF연계 증권거래계좌란 무엇인가요?",
            "expected_answer": "MMF연계 증권거래계좌는 예탁금을 MMF(단기금융집합투자기구)에 자동으로 투자하여 운용하는 계좌입니다. 일반 증권계좌와 달리 대기자금에 대해 MMF 수익을 얻을 수 있습니다. [참조: MMF연계 증권거래계좌 설정약관]",
            "category": "약관_CMA",
            "difficulty": "hard",
            "expected_sources": ["MMF연계 증권거래계좌 설정약관"]
        },
        {
            "id": "tos_033",
            "question": "QI투자제한동의서가 필요한 경우는 언제인가요?",
            "expected_answer": "QI(Qualified Intermediary)투자제한동의서는 미국 세법상 적격중개인 관련 투자 시 필요합니다. 미국 주식 등 해외증권 투자 시 관련 세금 처리를 위해 작성합니다. [참조: QI투자제한동의서]",
            "category": "약관_해외주식",
            "difficulty": "hard",
            "expected_sources": ["QI투자제한동의서"]
        },
        {
            "id": "tos_034",
            "question": "장내파생상품과 장외파생상품의 차이점은 무엇인가요?",
            "expected_answer": "장내파생상품은 거래소에서 표준화된 조건으로 거래되며 한국거래소가 계약이행을 보증합니다. 장외파생상품은 장외시장에서 맞춤형으로 거래되며 거래상대방 불이행 위험이 있습니다. [참조: EUREX 연계선물 설명서]",
            "category": "약관_파생상품",
            "difficulty": "hard",
            "expected_sources": ["EUREX 연계선물 위험고지서 및 거래 설명서"]
        },
        {
            "id": "tos_035",
            "question": "반대매매란 무엇인가요?",
            "expected_answer": "반대매매란 위탁증거금이 유지증거금에 미달하는 경우 고객 동의 없이 계약이 강제 청산되는 것을 말합니다. 증거금 상황을 수시로 확인하여 대응할 필요가 있습니다. [참조: 장내파생상품 위험고지서]",
            "category": "약관_파생상품",
            "difficulty": "medium",
            "expected_sources": ["EUREX 연계선물 위험고지서 및 거래 설명서"]
        },
    ]

    return tos_golden_set


def merge_with_existing_dataset(
    existing_path: str,
    tos_data: list[dict],
    output_path: str
) -> None:
    """기존 데이터셋과 ToS Golden set을 병합."""

    # 기존 데이터 로드
    with open(existing_path, 'r', encoding='utf-8') as f:
        existing_data = json.load(f)

    print(f"기존 데이터: {len(existing_data)}개")
    print(f"ToS 추가 데이터: {len(tos_data)}개")

    # 병합
    merged_data = existing_data + tos_data

    print(f"병합 후 총: {len(merged_data)}개")

    # 저장
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(merged_data, f, ensure_ascii=False, indent=4)

    print(f"저장 완료: {output_path}")

    # 카테고리 분포 출력
    categories = {}
    for item in merged_data:
        cat = item.get('category', 'N/A')
        categories[cat] = categories.get(cat, 0) + 1

    print("\n=== 최종 카테고리 분포 ===")
    for cat, count in sorted(categories.items(), key=lambda x: -x[1]):
        print(f"  {cat}: {count}개")


if __name__ == "__main__":
    # ToS Golden set 생성
    tos_golden_set = create_tos_golden_set()

    # 기존 데이터셋과 병합
    merge_with_existing_dataset(
        existing_path="data/evaluation/evaluation_dataset.json",
        tos_data=tos_golden_set,
        output_path="data/evaluation/evaluation_dataset_v2.json"
    )
