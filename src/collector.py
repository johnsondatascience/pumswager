"""Data collection orchestrator for PUMS data."""

import logging
from datetime import datetime
from typing import List, Optional

from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.orm import Session

from src.census_api import CensusAPIClient, STATE_CODES
from src.database import get_session
from src.models import CollectionJob, PumsHousehold, PumsPerson

logger = logging.getLogger(__name__)


class PumsDataCollector:
    """Orchestrates collection of PUMS data from Census API to database."""

    def __init__(self, api_key: Optional[str] = None, year: Optional[int] = None):
        """Initialize the collector.
        
        Args:
            api_key: Census API key.
            year: Survey year to collect.
        """
        self.client = CensusAPIClient(api_key=api_key, year=year)
        self.year = self.client.year

    def _create_job(
        self,
        session: Session,
        job_type: str,
        state_code: str,
    ) -> CollectionJob:
        """Create a new collection job record."""
        job = CollectionJob(
            job_type=job_type,
            state_code=state_code,
            year=self.year,
            status="running",
            started_at=datetime.utcnow(),
        )
        session.add(job)
        session.flush()
        return job

    def _update_job(
        self,
        session: Session,
        job: CollectionJob,
        status: str,
        records: int = 0,
        error: Optional[str] = None,
    ) -> None:
        """Update a collection job record."""
        job.status = status
        job.records_collected = records
        job.error_message = error
        if status in ("completed", "failed"):
            job.completed_at = datetime.utcnow()

    def collect_person_data(
        self,
        state_codes: Optional[List[str]] = None,
        batch_size: int = 1000,
    ) -> int:
        """Collect person-level PUMS data for specified states.
        
        Args:
            state_codes: List of FIPS state codes. If None, collects all states.
            batch_size: Number of records to insert per batch.
            
        Returns:
            Total number of records collected.
        """
        if state_codes is None:
            state_codes = list(STATE_CODES.keys())
        
        total_records = 0
        
        for state_code in state_codes:
            with get_session() as session:
                job = self._create_job(session, "person", state_code)
                
                try:
                    records = self.client.fetch_person_data(state_code)
                    
                    if records:
                        # Use upsert to handle duplicates
                        for i in range(0, len(records), batch_size):
                            batch = records[i:i + batch_size]
                            stmt = insert(PumsPerson).values(batch)
                            stmt = stmt.on_conflict_do_update(
                                constraint="uq_person_record",
                                set_={
                                    col.name: col
                                    for col in stmt.excluded
                                    if col.name not in ("id", "created_at", "serialno", "sporder", "year")
                                },
                            )
                            session.execute(stmt)
                        
                        total_records += len(records)
                        self._update_job(session, job, "completed", len(records))
                        logger.info(
                            f"Collected {len(records)} person records for state {state_code}"
                        )
                    else:
                        self._update_job(session, job, "completed", 0)
                        logger.warning(f"No person records found for state {state_code}")
                        
                except Exception as e:
                    self._update_job(session, job, "failed", error=str(e))
                    logger.error(f"Failed to collect person data for state {state_code}: {e}")
                    raise
        
        return total_records

    def collect_household_data(
        self,
        state_codes: Optional[List[str]] = None,
        batch_size: int = 1000,
    ) -> int:
        """Collect household-level PUMS data for specified states.
        
        Args:
            state_codes: List of FIPS state codes. If None, collects all states.
            batch_size: Number of records to insert per batch.
            
        Returns:
            Total number of records collected.
        """
        if state_codes is None:
            state_codes = list(STATE_CODES.keys())
        
        total_records = 0
        
        for state_code in state_codes:
            with get_session() as session:
                job = self._create_job(session, "household", state_code)
                
                try:
                    records = self.client.fetch_household_data(state_code)
                    
                    if records:
                        # Use upsert to handle duplicates
                        for i in range(0, len(records), batch_size):
                            batch = records[i:i + batch_size]
                            stmt = insert(PumsHousehold).values(batch)
                            stmt = stmt.on_conflict_do_update(
                                constraint="uq_household_record",
                                set_={
                                    col.name: col
                                    for col in stmt.excluded
                                    if col.name not in ("id", "created_at", "serialno", "year")
                                },
                            )
                            session.execute(stmt)
                        
                        total_records += len(records)
                        self._update_job(session, job, "completed", len(records))
                        logger.info(
                            f"Collected {len(records)} household records for state {state_code}"
                        )
                    else:
                        self._update_job(session, job, "completed", 0)
                        logger.warning(f"No household records found for state {state_code}")
                        
                except Exception as e:
                    self._update_job(session, job, "failed", error=str(e))
                    logger.error(f"Failed to collect household data for state {state_code}: {e}")
                    raise
        
        return total_records

    def collect_all(
        self,
        state_codes: Optional[List[str]] = None,
        batch_size: int = 1000,
    ) -> dict:
        """Collect both person and household data.
        
        Args:
            state_codes: List of FIPS state codes. If None, collects all states.
            batch_size: Number of records to insert per batch.
            
        Returns:
            Dictionary with counts of collected records.
        """
        logger.info(f"Starting full data collection for year {self.year}")
        
        person_count = self.collect_person_data(state_codes, batch_size)
        household_count = self.collect_household_data(state_codes, batch_size)
        
        logger.info(
            f"Collection complete: {person_count} person records, "
            f"{household_count} household records"
        )
        
        return {
            "person_records": person_count,
            "household_records": household_count,
            "year": self.year,
        }
