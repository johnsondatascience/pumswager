"""Main entry point for the ACS PUMS Data Collector."""

import argparse
import logging
import sys

from src.census_api import CensusAPIClient, STATE_CODES
from src.file_collector import PumsFileCollector
from src.config import census_config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger(__name__)


def list_states() -> None:
    """Print available state codes."""
    print("\nAvailable State Codes:")
    print("-" * 40)
    for code, name in sorted(STATE_CODES.items(), key=lambda x: x[1]):
        print(f"  {code}: {name}")
    print()


def check_connections() -> bool:
    """Check API connectivity."""
    print("\nChecking connections...")
    
    # Check Census API
    print("  Census API: ", end="")
    client = CensusAPIClient()
    if client.test_connection():
        print("OK")
    else:
        print("FAILED")
        if not census_config.api_key:
            print("    Warning: No API key configured. Get one at:")
            print("    https://api.census.gov/data/key_signup.html")
        return False
    
    print()
    return True


def collect_data(
    data_type: str,
    states: list,
    year: int,
    output_dir: str = "data",
) -> None:
    """Run data collection.
    
    Args:
        data_type: Type of data to collect ('person', 'household', or 'all').
        states: List of state codes to collect.
        year: Survey year.
        output_dir: Directory to write output files.
    """
    collector = PumsFileCollector(year=year, output_dir=output_dir)
    
    state_codes = states if states else None
    state_desc = ", ".join(states) if states else "all states"
    
    print(f"\nCollecting {data_type} data for {state_desc}, year {year}...")
    print(f"Output directory: {output_dir}")
    print("-" * 60)
    
    try:
        if data_type == "person":
            count = collector.collect_person_data(state_codes)
            print(f"\nCompleted: {count} person records written to {output_dir}/")
        elif data_type == "household":
            count = collector.collect_household_data(state_codes)
            print(f"\nCompleted: {count} household records written to {output_dir}/")
        else:  # all
            results = collector.collect_all(state_codes)
            print(f"\nCompleted:")
            print(f"  Person records: {results['person_records']}")
            print(f"  Household records: {results['household_records']}")
            print(f"  Output: {results['output_dir']}/")
    except Exception as e:
        logger.error(f"Collection failed: {e}")
        sys.exit(1)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="ACS PUMS Data Collector - Fetch Census Bureau microdata",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Check connectivity
  python main.py --check
  
  # List available state codes
  python main.py --list-states
  
  # Collect all data for California
  python main.py --collect all --states 06
  
  # Collect person data for multiple states
  python main.py --collect person --states 06 36 48
  
  # Collect household data for all states
  python main.py --collect household
  
  # Specify a different year
  python main.py --collect all --year 2021 --states 06
        """,
    )
    
    parser.add_argument(
        "--check",
        action="store_true",
        help="Check API connectivity",
    )
    parser.add_argument(
        "--list-states",
        action="store_true",
        help="List available state codes",
    )
    parser.add_argument(
        "--collect",
        choices=["person", "household", "all"],
        help="Type of data to collect",
    )
    parser.add_argument(
        "--states",
        nargs="+",
        help="State codes to collect (e.g., 06 36 48). If not specified, collects all states.",
    )
    parser.add_argument(
        "--year",
        type=int,
        default=census_config.year,
        help=f"Survey year (default: {census_config.year})",
    )
    parser.add_argument(
        "--output",
        default="data",
        help="Output directory for CSV files (default: data)",
    )
    
    args = parser.parse_args()
    
    # Handle commands
    if args.list_states:
        list_states()
        return
    
    if args.check:
        success = check_connections()
        sys.exit(0 if success else 1)
    
    if args.collect:
        # Validate state codes
        if args.states:
            invalid = [s for s in args.states if s not in STATE_CODES]
            if invalid:
                print(f"Error: Invalid state codes: {', '.join(invalid)}")
                print("Use --list-states to see valid codes.")
                sys.exit(1)
        
        # Check connections before collecting
        if not check_connections():
            print("Please fix connection issues before collecting data.")
            sys.exit(1)
        
        collect_data(args.collect, args.states, args.year, args.output)
        return
    
    # No command specified
    parser.print_help()


if __name__ == "__main__":
    main()
