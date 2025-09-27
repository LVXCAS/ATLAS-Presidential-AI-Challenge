#!/usr/bin/env python3
"""
MISSING SYSTEMS ANALYZER
Identifies what additional systems should be running for maximum ROI
Ensures comprehensive coverage for explosive opportunities
"""

import os
import glob
from datetime import datetime

class MissingSystemsAnalyzer:
    """Analyzes what systems should be running but aren't"""

    def __init__(self):
        # Critical systems that should always be running
        self.critical_systems = {
            'overnight_gap_scanner.py': {
                'purpose': 'Catch pre-market explosive gaps',
                'roi_type': 'Overnight surprises, earnings gaps',
                'priority': 'CRITICAL'
            },
            'options_greeks_monitor.py': {
                'purpose': 'Monitor options Greeks changes for explosive setups',
                'roi_type': 'Options leverage opportunities',
                'priority': 'HIGH'
            },
            'truly_autonomous_trader.py': {
                'purpose': 'Fully autonomous trading execution',
                'roi_type': 'Real-time opportunity capture',
                'priority': 'HIGH'
            },
            'master_profit_maximization_system.py': {
                'purpose': 'Coordinate all profit maximization',
                'roi_type': 'System-wide optimization',
                'priority': 'CRITICAL'
            },
            'intelligent_rebalancer.py': {
                'purpose': 'Dynamic portfolio rebalancing',
                'roi_type': 'Position optimization',
                'priority': 'MEDIUM'
            },
            'live_capital_allocation_engine.py': {
                'purpose': 'Real-time capital allocation',
                'roi_type': 'Optimal capital deployment',
                'priority': 'HIGH'
            }
        }

        # Specialized ROI hunting systems
        self.roi_hunting_systems = {
            'options_decision_advisor.py': {
                'purpose': 'Options decision optimization',
                'roi_type': 'Options strategy selection',
                'priority': 'HIGH'
            },
            'precise_buying_power_trader.py': {
                'purpose': 'Maximize buying power utilization',
                'roi_type': 'Capital efficiency',
                'priority': 'MEDIUM'
            },
            'quality_constrained_trader.py': {
                'purpose': 'Quality-focused trading',
                'roi_type': 'Institutional-grade opportunities',
                'priority': 'HIGH'
            }
        }

    def check_system_files(self):
        """Check which system files exist"""

        print("MISSING SYSTEMS ANALYSIS")
        print("=" * 60)
        print("Identifying systems that should be running for maximum ROI")
        print("=" * 60)

        existing_files = []
        missing_files = []

        all_systems = {**self.critical_systems, **self.roi_hunting_systems}

        print(f"\n=== SYSTEM FILE CHECK ===")
        print("System | Status | Priority | Purpose")
        print("-" * 80)

        for filename, info in all_systems.items():
            if os.path.exists(filename):
                status = "EXISTS"
                existing_files.append(filename)
            else:
                status = "MISSING"
                missing_files.append(filename)

            print(f"{filename:<35} | {status:<7} | {info['priority']:<8} | {info['purpose']}")

        print("-" * 80)
        print(f"EXISTING: {len(existing_files)}")
        print(f"MISSING: {len(missing_files)}")

        return existing_files, missing_files

    def analyze_roi_gaps(self, missing_files):
        """Analyze what ROI opportunities we're missing"""

        if not missing_files:
            print(f"\n✅ ALL SYSTEMS PRESENT - No ROI gaps detected")
            return

        print(f"\n=== ROI OPPORTUNITY GAPS ===")
        print("Missing systems and the ROI opportunities they would capture:")
        print("-" * 60)

        all_systems = {**self.critical_systems, **self.roi_hunting_systems}

        critical_missing = []
        high_missing = []

        for filename in missing_files:
            if filename in all_systems:
                info = all_systems[filename]
                if info['priority'] == 'CRITICAL':
                    critical_missing.append((filename, info))
                elif info['priority'] == 'HIGH':
                    high_missing.append((filename, info))

        if critical_missing:
            print("🚨 CRITICAL MISSING (Major ROI Impact):")
            for filename, info in critical_missing:
                print(f"   {filename}")
                print(f"      ROI Type: {info['roi_type']}")
                print(f"      Impact: {info['purpose']}")
                print()

        if high_missing:
            print("⚠️ HIGH PRIORITY MISSING (Significant ROI Impact):")
            for filename, info in high_missing:
                print(f"   {filename}")
                print(f"      ROI Type: {info['roi_type']}")
                print(f"      Impact: {info['purpose']}")
                print()

    def recommend_additional_systems(self):
        """Recommend additional systems to run"""

        print(f"\n=== RECOMMENDED ADDITIONAL SYSTEMS ===")
        print("Systems that would enhance explosive ROI detection:")
        print("-" * 60)

        additional_systems = [
            {
                'name': 'Catalyst Calendar Monitor',
                'purpose': 'Track earnings, FDA approvals, economic events',
                'roi_potential': 'High - catalyst-driven explosive moves',
                'implementation': 'Calendar scraping + alert system'
            },
            {
                'name': 'Unusual Options Activity Scanner',
                'purpose': 'Detect unusual options volume/open interest',
                'roi_potential': 'Very High - insider/institutional activity',
                'implementation': 'Options chain analysis + volume alerts'
            },
            {
                'name': 'After Hours Momentum Tracker',
                'purpose': 'Track after-hours price/volume surges',
                'roi_potential': 'High - early momentum detection',
                'implementation': 'Extended hours monitoring'
            },
            {
                'name': 'Social Sentiment Spike Detector',
                'purpose': 'Detect viral stock mentions and sentiment shifts',
                'roi_potential': 'Medium-High - meme stock potential',
                'implementation': 'Social media API monitoring'
            },
            {
                'name': 'Institutional Flow Monitor',
                'purpose': 'Track large institutional buying/selling',
                'roi_potential': 'High - follow smart money',
                'implementation': 'Block trade and dark pool analysis'
            }
        ]

        for i, system in enumerate(additional_systems, 1):
            print(f"{i}. {system['name']}")
            print(f"   Purpose: {system['purpose']}")
            print(f"   ROI Potential: {system['roi_potential']}")
            print(f"   Implementation: {system['implementation']}")
            print()

    def generate_startup_sequence(self, existing_files):
        """Generate recommended startup sequence for missing systems"""

        print(f"\n=== STARTUP SEQUENCE RECOMMENDATION ===")
        print("Order to start systems for maximum ROI impact:")
        print("-" * 60)

        startup_sequence = []

        # Add critical systems first
        for filename, info in self.critical_systems.items():
            if filename in existing_files:
                startup_sequence.append({
                    'file': filename,
                    'priority': info['priority'],
                    'purpose': info['purpose'],
                    'action': 'START_IF_NOT_RUNNING'
                })

        # Add high priority ROI systems
        for filename, info in self.roi_hunting_systems.items():
            if filename in existing_files and info['priority'] == 'HIGH':
                startup_sequence.append({
                    'file': filename,
                    'priority': info['priority'],
                    'purpose': info['purpose'],
                    'action': 'START_IF_NOT_RUNNING'
                })

        print("RECOMMENDED STARTUP ORDER:")
        for i, system in enumerate(startup_sequence, 1):
            print(f"{i:2d}. {system['file']:<35} ({system['priority']:<8}) - {system['purpose']}")

        return startup_sequence

    def run_analysis(self):
        """Run complete missing systems analysis"""

        existing_files, missing_files = self.check_system_files()

        self.analyze_roi_gaps(missing_files)

        self.recommend_additional_systems()

        startup_sequence = self.generate_startup_sequence(existing_files)

        print("\n" + "=" * 60)
        print("ANALYSIS COMPLETE")
        print(f"Ready to maximize ROI with {len(existing_files)} available systems")
        print("=" * 60)

        return {
            'existing_systems': existing_files,
            'missing_systems': missing_files,
            'startup_sequence': startup_sequence
        }

def main():
    """Run missing systems analysis"""
    analyzer = MissingSystemsAnalyzer()
    results = analyzer.run_analysis()
    return results

if __name__ == "__main__":
    main()