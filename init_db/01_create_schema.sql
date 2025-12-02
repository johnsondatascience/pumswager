-- ACS PUMS Database Schema

-- Table for storing person-level PUMS records
CREATE TABLE IF NOT EXISTS pums_person (
    id SERIAL PRIMARY KEY,
    serialno VARCHAR(20) NOT NULL,          -- Housing unit/GQ person serial number
    sporder INTEGER,                         -- Person number within household
    puma VARCHAR(10),                        -- Public Use Microdata Area code
    st VARCHAR(2),                           -- State code
    year INTEGER NOT NULL,                   -- Survey year
    
    -- Demographics
    agep INTEGER,                            -- Age
    sex INTEGER,                             -- Sex (1=Male, 2=Female)
    rac1p INTEGER,                           -- Race code
    hisp INTEGER,                            -- Hispanic origin
    nativity INTEGER,                        -- Nativity (1=Native, 2=Foreign born)
    cit INTEGER,                             -- Citizenship status
    
    -- Education
    schl INTEGER,                            -- Educational attainment
    
    -- Employment
    esr INTEGER,                             -- Employment status
    cow INTEGER,                             -- Class of worker
    occp VARCHAR(10),                        -- Occupation code
    indp VARCHAR(10),                        -- Industry code
    wkhp INTEGER,                            -- Hours worked per week
    wkw INTEGER,                             -- Weeks worked in past 12 months
    
    -- Income
    wagp INTEGER,                            -- Wages/salary income
    semp INTEGER,                            -- Self-employment income
    pincp INTEGER,                           -- Total person income
    pap INTEGER,                             -- Public assistance income
    retp INTEGER,                            -- Retirement income
    ssip INTEGER,                            -- Supplemental Security Income
    ssp INTEGER,                             -- Social Security income
    
    -- Weight
    pwgtp INTEGER,                           -- Person weight
    
    -- Metadata
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    UNIQUE(serialno, sporder, year)
);

-- Table for storing household-level PUMS records
CREATE TABLE IF NOT EXISTS pums_household (
    id SERIAL PRIMARY KEY,
    serialno VARCHAR(20) NOT NULL,          -- Housing unit serial number
    puma VARCHAR(10),                        -- Public Use Microdata Area code
    st VARCHAR(2),                           -- State code
    year INTEGER NOT NULL,                   -- Survey year
    
    -- Housing characteristics
    type_hu INTEGER,                         -- Type of unit
    bld INTEGER,                             -- Building type
    ten INTEGER,                             -- Tenure (own/rent)
    rmsp INTEGER,                            -- Number of rooms
    bdsp INTEGER,                            -- Number of bedrooms
    ybl INTEGER,                             -- Year building built
    
    -- Household composition
    np INTEGER,                              -- Number of persons
    hht INTEGER,                             -- Household type
    hht2 INTEGER,                            -- Detailed household type (2023+)
    hupaoc INTEGER,                          -- Presence of children
    huparc INTEGER,                          -- Presence of related children (deprecated 2023)
    
    -- Income
    hincp INTEGER,                           -- Household income
    fincp INTEGER,                           -- Family income
    
    -- Costs
    grntp INTEGER,                           -- Gross rent
    smocp INTEGER,                           -- Selected monthly owner costs
    grpip INTEGER,                           -- Gross rent as % of income
    ocpip INTEGER,                           -- Owner costs as % of income
    
    -- Weight
    wgtp INTEGER,                            -- Housing unit weight
    
    -- Metadata
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    UNIQUE(serialno, year)
);

-- Table for tracking data collection jobs
CREATE TABLE IF NOT EXISTS collection_jobs (
    id SERIAL PRIMARY KEY,
    job_type VARCHAR(50) NOT NULL,           -- 'person' or 'household'
    state_code VARCHAR(2),                   -- State being collected
    year INTEGER NOT NULL,
    status VARCHAR(20) DEFAULT 'pending',    -- pending, running, completed, failed
    records_collected INTEGER DEFAULT 0,
    error_message TEXT,
    started_at TIMESTAMP,
    completed_at TIMESTAMP,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Indexes for common queries
CREATE INDEX IF NOT EXISTS idx_person_state_year ON pums_person(st, year);
CREATE INDEX IF NOT EXISTS idx_person_puma ON pums_person(puma);
CREATE INDEX IF NOT EXISTS idx_person_serialno ON pums_person(serialno);
CREATE INDEX IF NOT EXISTS idx_household_state_year ON pums_household(st, year);
CREATE INDEX IF NOT EXISTS idx_household_puma ON pums_household(puma);
CREATE INDEX IF NOT EXISTS idx_household_serialno ON pums_household(serialno);
CREATE INDEX IF NOT EXISTS idx_jobs_status ON collection_jobs(status);
