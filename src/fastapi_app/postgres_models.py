from __future__ import annotations

from dataclasses import asdict

from pgvector.sqlalchemy import Vector
from sqlalchemy import Index
from sqlalchemy.orm import DeclarativeBase, Mapped, MappedAsDataclass, mapped_column


# Define the models
class Base(DeclarativeBase, MappedAsDataclass):
    pass


class Item(Base):
    __tablename__ = "packages"
    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    package_name: Mapped[str] = mapped_column()
    package_picture: Mapped[str] = mapped_column()
    url: Mapped[str] = mapped_column()
    price: Mapped[float] = mapped_column()
    cash_discount: Mapped[float] = mapped_column()
    installment_month: Mapped[str] = mapped_column()
    installment_limit: Mapped[str] = mapped_column()
    price_to_reserve_for_this_package: Mapped[str] = mapped_column()
    shop_name: Mapped[str] = mapped_column()
    category: Mapped[str] = mapped_column()
    category_tags: Mapped[str] = mapped_column()
    preview_1_10: Mapped[str] = mapped_column()
    selling_point: Mapped[str] = mapped_column()
    meta_keywords: Mapped[str] = mapped_column()
    brand: Mapped[str] = mapped_column()
    min_max_age: Mapped[str] = mapped_column()
    locations_time_open_close_how_to_transport_parking_google_maps: Mapped[str] = mapped_column()
    meta_description: Mapped[str] = mapped_column()
    price_details: Mapped[str] = mapped_column()
    package_details: Mapped[str] = mapped_column()
    important_info: Mapped[str] = mapped_column()
    payment_booking_info: Mapped[str] = mapped_column()
    general_info: Mapped[str] = mapped_column()
    early_signs_for_diagnosis: Mapped[str] = mapped_column()
    how_to_diagnose: Mapped[str] = mapped_column()
    hdcare_summary: Mapped[str] = mapped_column()
    common_question: Mapped[str] = mapped_column()
    know_this_disease: Mapped[str] = mapped_column()
    courses_of_action: Mapped[str] = mapped_column()
    signals_to_proceed_surgery: Mapped[str] = mapped_column()
    get_to_know_this_surgery: Mapped[str] = mapped_column()
    comparisons: Mapped[str] = mapped_column()
    getting_ready: Mapped[str] = mapped_column()
    recovery: Mapped[str] = mapped_column()
    side_effects: Mapped[str] = mapped_column()
    review_4_5_stars: Mapped[str] = mapped_column()
    brand_option_in_thai_name: Mapped[str] = mapped_column()
    brand_ranking_position: Mapped[int] = mapped_column()
    faq: Mapped[str] = mapped_column()
    embedding: Mapped[Vector] = mapped_column(Vector(1536))  # ada-002

    def to_dict(self, include_embedding: bool = False):
        model_dict = asdict(self)
        if include_embedding:
            model_dict["embedding"] = model_dict["embedding"].tolist()
        else:
            del model_dict["embedding"]
        return model_dict

    def to_str_for_rag(self):
        return f"""package_name: {self.package_name}
package_picture: {self.package_picture}
url: {self.url}
price: {self.price}
cash_discount: {self.cash_discount}
installment_month: {self.installment_month}
installment_limit: {self.installment_limit}
price_to_reserve_for_this_package: {self.price_to_reserve_for_this_package}
brand: {self.brand}
price_details: {self.price_details}
package_details: {self.package_details}
important_info: {self.important_info}
payment_booking_info: {self.payment_booking_info}
general_info: {self.general_info}
brand_ranking_position: {self.brand_ranking_position}
faq: {self.faq}
"""

    def to_str_for_embedding(self):
        return f"Name: {self.package_name} Description: {self.package_details}"


# Define HNSW index to support vector similarity search through the vector_cosine_ops access method (cosine distance).
index = Index(
    "hnsw_index_for_innerproduct_item_embedding",
    Item.embedding,
    postgresql_using="hnsw",
    postgresql_with={"m": 16, "ef_construction": 64},
    postgresql_ops={"embedding": "vector_ip_ops"},
)
