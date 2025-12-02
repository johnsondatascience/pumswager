"""Data collection to CSV files for PUMS data."""

import csv
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import List, Optional

from src.census_api import CensusAPIClient, STATE_CODES

logger = logging.getLogger(__name__)


class PumsFileCollector:
    """Collects PUMS data from Census API and writes to CSV files."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        year: Optional[int] = None,
        output_dir: str = "data",
    ):
        """Initialize the collector.
        
        Args:
            api_key: Census API key.
            year: Survey year to collect.
            output_dir: Directory to write CSV files.
        """
        self.client = CensusAPIClient(api_key=api_key, year=year)
        self.year = self.client.year
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _write_csv(self, records: List[dict], filename: str) -> str:
        """Write records to a CSV file.
        
        Args:
            records: List of dictionaries to write.
            filename: Name of the output file.
            
        Returns:
            Path to the written file.
        """
        if not records:
            return ""
        
        filepath = self.output_dir / filename
        fieldnames = list(records[0].keys())
        
        with open(filepath, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(records)
        
        return str(filepath)

    def collect_person_data(
        self,
        state_codes: Optional[List[str]] = None,
    ) -> int:
        """Collect person-level PUMS data for specified states.
        
        Args:
            state_codes: List of FIPS state codes. If None, collects all states.
            
        Returns:
            Total number of records collected.
        """
        if state_codes is None:
            state_codes = list(STATE_CODES.keys())
        
        total_records = 0
        all_records = []
        
        for state_code in state_codes:
            try:
                logger.info(f"Fetching person data for state {state_code}...")
                records = self.client.fetch_person_data(state_code)
                
                if records:
                    all_records.extend(records)
                    total_records += len(records)
                    logger.info(f"Fetched {len(records)} person records for state {state_code}")
                else:
                    logger.warning(f"No person records found for state {state_code}")
                    
            except Exception as e:
                logger.error(f"Failed to collect person data for state {state_code}: {e}")
                logger.warning(f"Skipping state {state_code}, continuing with remaining states...")
                continue
        
        # Write all records to CSV
        if all_records:
            filename = f"pums_person_{self.year}.csv"
            filepath = self._write_csv(all_records, filename)
            logger.info(f"Wrote {total_records} person records to {filepath}")
        
        return total_records

    def collect_household_data(
        self,
        state_codes: Optional[List[str]] = None,
    ) -> int:
        """Collect household-level PUMS data for specified states.
        
        Args:
            state_codes: List of FIPS state codes. If None, collects all states.
            
        Returns:
            Total number of records collected.
        """
        if state_codes is None:
            state_codes = list(STATE_CODES.keys())
        
        total_records = 0
        all_records = []
        
        for state_code in state_codes:
            try:
                logger.info(f"Fetching household data for state {state_code}...")
                records = self.client.fetch_household_data(state_code)
                
                if records:
                    all_records.extend(records)
                    total_records += len(records)
                    logger.info(f"Fetched {len(records)} household records for state {state_code}")
                else:
                    logger.warning(f"No household records found for state {state_code}")
                    
            except Exception as e:
                logger.error(f"Failed to collect household data for state {state_code}: {e}")
                logger.warning(f"Skipping state {state_code}, continuing with remaining states...")
                continue
        
        # Write all records to CSV
        if all_records:
            filename = f"pums_household_{self.year}.csv"
            filepath = self._write_csv(all_records, filename)
            logger.info(f"Wrote {total_records} household records to {filepath}")
        
        return total_records

    def collect_all(
        self,
        state_codes: Optional[List[str]] = None,
    ) -> dict:
        """Collect both person and household data.
        
        Args:
            state_codes: List of FIPS state codes. If None, collects all states.
            
        Returns:
            Dictionary with counts of collected records.
        """
        logger.info(f"Starting full data collection for year {self.year}")
        
        person_count = self.collect_person_data(state_codes)
        household_count = self.collect_household_data(state_codes)
        
        logger.info(
            f"Collection complete: {person_count} person records, "
            f"{household_count} household records"
        )
        
        return {
            "person_records": person_count,
            "household_records": household_count,
            "year": self.year,
            "output_dir": str(self.output_dir),
        }
