import argparse
import asyncio
import logging
import os

import numpy as np
import pandas as pd
import sqlalchemy.exc
from dotenv import load_dotenv
from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import async_sessionmaker
from tqdm import tqdm

from fastapi_app.postgres_engine import (
    create_postgres_engine_from_args,
    create_postgres_engine_from_env,
)
from fastapi_app.postgres_models import Item

logger = logging.getLogger("ragapp")

embedding_fields = [
    'embedding_package_name', 'embedding_package_picture', 'embedding_url', 
    'embedding_installment_month', 'embedding_installment_limit', 
    'embedding_price_to_reserve_for_this_package', 'embedding_shop_name', 
    'embedding_category', 'embedding_category_tags', 'embedding_preview_1_10', 
    'embedding_selling_point', 'embedding_meta_keywords', 'embedding_brand', 
    'embedding_min_max_age', 'embedding_locations', 'embedding_meta_description', 
    'embedding_price_details', 'embedding_package_details', 'embedding_important_info', 
    'embedding_payment_booking_info', 'embedding_general_info', 'embedding_early_signs_for_diagnosis', 
    'embedding_how_to_diagnose', 'embedding_hdcare_summary', 'embedding_common_question', 
    'embedding_know_this_disease', 'embedding_courses_of_action', 'embedding_signals_to_proceed_surgery', 
    'embedding_get_to_know_this_surgery', 'embedding_comparisons', 'embedding_getting_ready', 
    'embedding_recovery', 'embedding_side_effects', 'embedding_review_4_5_stars', 
    'embedding_brand_option_in_thai_name', 'embedding_faq'
]

def convert_to_int(value):
    try:
        return int(value)
    except (ValueError, TypeError):
        return None

def convert_to_float(value):
    try:
        return float(value)
    except (ValueError, TypeError):
        return None

async def seed_data(engine):
    logger.info("Checking if the packages table exists...")
    async with engine.begin() as conn:
        result = await conn.execute(
            text(
                """
                SELECT EXISTS 
                (SELECT 1 FROM information_schema.tables WHERE table_schema = 'public' AND table_name = 'packages')
                """  # noqa
            )
        )
        if not result.scalar():
            logger.error("Packages table does not exist. Please run the database setup script first.")
            return

    async with async_sessionmaker(engine, expire_on_commit=False)() as session:
        # Read the CSV file
        current_dir = os.path.dirname(os.path.realpath(__file__))
        csv_path = (
            os.path.join(current_dir, "packages.csv")
        )

        try:
            df = pd.read_csv(csv_path, delimiter=',', quotechar='"', escapechar='\\', on_bad_lines='skip', encoding='utf-8')
        except pd.errors.ParserError as e:
            logger.error(f"Error reading CSV file: {e}")
            return

        # Replace NaN values in string columns with None
        str_columns = df.select_dtypes(include=[object]).columns
        df[str_columns] = df[str_columns].replace({np.nan: None})

        # Replace NaN values in numeric columns with None
        num_columns = df.select_dtypes(include=[np.number]).columns
        df[num_columns] = df[num_columns].replace({np.nan: None})

        # Convert DataFrame to list of dictionaries
        records = df.to_dict(orient='records')

        logger.info("Starting to insert records into the database...")
        # Insert records into the database with progress bar
        for record in tqdm(records, desc="Inserting records"):
            try:
                # Ensure the id is cast to integer
                record["id"] = convert_to_int(record["id"])
                if record["id"] is None:
                    logger.error(f"Skipping record with invalid id: {record}")
                    continue
                
                # Explicitly cast other fields to the appropriate types
                if "price" in record:
                    record["price"] = convert_to_float(record["price"])
                if "cash_discount" in record:
                    record["cash_discount"] = convert_to_float(record["cash_discount"])
                if "brand_ranking_position" in record:
                    record["brand_ranking_position"] = convert_to_int(record["brand_ranking_position"])

                # Skip record if price or cash_discount cannot be converted to float
                if record["price"] is None or record["cash_discount"] is None:
                    logger.error(f"Skipping record with invalid numeric fields: {record}")
                    continue

                item = await session.execute(select(Item).filter(Item.id == record["id"]))
                if item.scalars().first():
                    continue

                # Create a new Item instance without embedding columns
                item_data = {key: value for key, value in record.items() if key in Item.__table__.columns}
                
                # Set embedding fields to None
                for field in embedding_fields:
                    item_data[field] = None
                
                item = Item(**item_data)
                session.add(item)

            except Exception as e:
                logger.error(f"Error inserting record with id {record['id']}: {e}")
                await session.rollback()
                continue

        try:
            await session.commit()
            logger.info("All records inserted successfully.")
        except sqlalchemy.exc.IntegrityError as e:
            logger.error(f"Integrity error during commit: {e}")

async def main():
    parser = argparse.ArgumentParser(description="Seed database with CSV data")
    parser.add_argument("--host", type=str, help="Postgres host")
    parser.add_argument("--username", type=str, help="Postgres username")
    parser.add_argument("--password", type=str, help="Postgres password")
    parser.add_argument("--database", type=str, help="Postgres database")
    parser.add_argument("--sslmode", type=str, help="Postgres sslmode")

    # if no args are specified, use environment variables
    args = parser.parse_args()
    if args.host is None:
        engine = await create_postgres_engine_from_env()
    else:
        engine = await create_postgres_engine_from_args(args)

    await seed_data(engine)

    await engine.dispose()

if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)
    logger.setLevel(logging.INFO)
    load_dotenv(override=True)
    asyncio.run(main())
