"""
REF 제품명 파싱 결과 모델.
Java의 REFParsedProductName VO에 대응.
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class REFParsedProductName:
    """파싱된 제품명 정보를 담는 Value Object."""

    original_text: Optional[str] = None
    """원본 제품명"""

    refined_text: Optional[str] = None
    """다음 프로세스로 보낼 최종 텍스트 (keep 옵션에 따라 브랜드/용량 재부착 결과)"""

    core_text: Optional[str] = None
    """순수 정제 제품명 (브랜드·용량·수량·노이즈 전부 제거한 코어)"""

    brand_name: Optional[str] = None
    """추출된 브랜드명"""

    quantity: Optional[int] = None
    """수량 (예: 3개입 → 3)"""

    volume: Optional[str] = None
    """용량 문자열 (예: '500ml')"""

    volume_unit: Optional[str] = None
    """용량 단위 (예: 'ml', 'g', 'kg', 'l')"""

    def __repr__(self) -> str:
        return (
            f"REFParsedProductName("
            f"original='{self.original_text}', "
            f"refined='{self.refined_text}', "
            f"core='{self.core_text}', "
            f"brand='{self.brand_name}', "
            f"quantity={self.quantity}, "
            f"volume='{self.volume}')"
        )
