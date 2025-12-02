"""SQLAlchemy ORM models for PUMS data."""

from datetime import datetime
from typing import Optional

from sqlalchemy import Column, Integer, String, DateTime, Text, UniqueConstraint
from sqlalchemy.orm import declarative_base

Base = declarative_base()


class PumsPerson(Base):
    """Person-level PUMS record."""
    
    __tablename__ = "pums_person"
    __table_args__ = (
        UniqueConstraint("serialno", "sporder", "year", name="uq_person_record"),
    )

    id = Column(Integer, primary_key=True)
    serialno = Column(String(20), nullable=False)
    sporder = Column(Integer)
    puma = Column(String(10))
    st = Column(String(2))
    year = Column(Integer, nullable=False)

    # Demographics
    agep = Column(Integer)
    sex = Column(Integer)
    rac1p = Column(Integer)
    hisp = Column(Integer)
    nativity = Column(Integer)
    cit = Column(Integer)

    # Education
    schl = Column(Integer)

    # Employment
    esr = Column(Integer)
    cow = Column(Integer)
    occp = Column(String(10))
    indp = Column(String(10))
    wkhp = Column(Integer)
    wkw = Column(Integer)

    # Income
    wagp = Column(Integer)
    semp = Column(Integer)
    pincp = Column(Integer)
    pap = Column(Integer)
    retp = Column(Integer)
    ssip = Column(Integer)
    ssp = Column(Integer)

    # Weight
    pwgtp = Column(Integer)

    created_at = Column(DateTime, default=datetime.utcnow)


class PumsHousehold(Base):
    """Household-level PUMS record."""
    
    __tablename__ = "pums_household"
    __table_args__ = (
        UniqueConstraint("serialno", "year", name="uq_household_record"),
    )

    id = Column(Integer, primary_key=True)
    serialno = Column(String(20), nullable=False)
    puma = Column(String(10))
    st = Column(String(2))
    year = Column(Integer, nullable=False)

    # Housing characteristics
    type_hu = Column(Integer)
    bld = Column(Integer)
    ten = Column(Integer)
    rmsp = Column(Integer)
    bdsp = Column(Integer)
    ybl = Column(Integer)

    # Household composition
    np = Column(Integer)
    hht = Column(Integer)
    hht2 = Column(Integer)  # Detailed household type (2023+)
    hupaoc = Column(Integer)
    huparc = Column(Integer)  # Deprecated in 2023, kept for backward compatibility

    # Income
    hincp = Column(Integer)
    fincp = Column(Integer)

    # Costs
    grntp = Column(Integer)
    smocp = Column(Integer)
    grpip = Column(Integer)
    ocpip = Column(Integer)

    # Weight
    wgtp = Column(Integer)

    created_at = Column(DateTime, default=datetime.utcnow)


class CollectionJob(Base):
    """Track data collection jobs."""
    
    __tablename__ = "collection_jobs"

    id = Column(Integer, primary_key=True)
    job_type = Column(String(50), nullable=False)
    state_code = Column(String(2))
    year = Column(Integer, nullable=False)
    status = Column(String(20), default="pending")
    records_collected = Column(Integer, default=0)
    error_message = Column(Text)
    started_at = Column(DateTime)
    completed_at = Column(DateTime)
    created_at = Column(DateTime, default=datetime.utcnow)
