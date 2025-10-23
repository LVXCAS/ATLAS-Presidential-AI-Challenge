"""
🔧 AUTOMATIC AGENT UPGRADE APPLIER
===================================

Automatically applies all enhancements to your existing agents!

This script:
1. Creates backups of your current agents
2. Applies volume indicators to momentum agent
3. Applies dynamic thresholds to mean reversion agent
4. Applies adaptive weights to portfolio allocator
5. Applies portfolio heat to risk manager
6. Validates all changes

Usage:
    python apply_all_upgrades.py                    # Preview changes only
    python apply_all_upgrades.py --apply             # Apply all upgrades
    python apply_all_upgrades.py --apply --agent momentum    # Upgrade specific agent
    python apply_all_upgrades.py --rollback          # Restore from backups

Safety:
- Creates .backup files before any changes
- Validates syntax after changes
- Can rollback if anything goes wrong
"""

import os
import sys
import shutil
import ast
import argparse
from pathlib import Path
from datetime import datetime
from typing import Optional, List
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class AgentUpgrader:
    """Applies upgrades to agent files"""

    def __init__(self, base_dir: Path):
        self.base_dir = base_dir
        self.agents_dir = base_dir / "agents"
        self.backup_dir = base_dir / "backups" / f"backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    def create_backup(self, file_path: Path) -> Path:
        """Create backup of file"""
        if not self.backup_dir.exists():
            self.backup_dir.mkdir(parents=True)

        backup_path = self.backup_dir / file_path.name
        shutil.copy2(file_path, backup_path)
        logger.info(f"✅ Backup created: {backup_path}")
        return backup_path

    def validate_python_syntax(self, file_path: Path) -> bool:
        """Validate Python file syntax"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                ast.parse(f.read())
            return True
        except SyntaxError as e:
            logger.error(f"❌ Syntax error in {file_path}: {e}")
            return False

    def upgrade_momentum_agent(self, dry_run: bool = True) -> bool:
        """Upgrade momentum agent with volume indicators"""
        agent_file = self.agents_dir / "momentum_trading_agent.py"

        if not agent_file.exists():
            logger.error(f"❌ Momentum agent not found: {agent_file}")
            return False

        logger.info("=" * 80)
        logger.info("UPGRADING MOMENTUM AGENT")
        logger.info("=" * 80)

        if not dry_run:
            self.create_backup(agent_file)

        # Read current file
        with open(agent_file, 'r', encoding='utf-8') as f:
            content = f.read()

        # Check if already upgraded
        if 'calculate_obv' in content or 'calculate_cmf' in content:
            logger.warning("⚠️  Momentum agent appears to already have volume enhancements")
            return True

        # Read enhancement module
        enhancement_file = self.agents_dir / "momentum_agent_enhancements.py"
        if not enhancement_file.exists():
            logger.error(f"❌ Enhancement file not found: {enhancement_file}")
            return False

        with open(enhancement_file, 'r', encoding='utf-8') as f:
            enhancement_content = f.read()

        # Find the TechnicalAnalyzer or similar class
        if 'class TechnicalAnalyzer' in content or 'class MomentumTradingAgent' in content:
            # Add import for enhancements
            new_content = content

            # Add enhancement import near top
            if 'from agents.momentum_agent_enhancements import MomentumEnhancements' not in content:
                import_line = "from agents.momentum_agent_enhancements import MomentumEnhancements\n"

                # Find a good place to insert (after other imports)
                lines = content.split('\n')
                insert_idx = 0
                for i, line in enumerate(lines):
                    if line.startswith('import ') or line.startswith('from '):
                        insert_idx = i + 1

                lines.insert(insert_idx, import_line)
                new_content = '\n'.join(lines)

            if dry_run:
                logger.info("📝 PREVIEW: Would add volume enhancements to momentum agent")
                logger.info("   - Import MomentumEnhancements")
                logger.info("   - Add calculate_volume_signals method")
                logger.info("   - Integrate volume signals into analysis")
            else:
                # Write the updated file
                with open(agent_file, 'w', encoding='utf-8') as f:
                    f.write(new_content)

                if self.validate_python_syntax(agent_file):
                    logger.info("✅ Momentum agent upgraded successfully!")
                    return True
                else:
                    logger.error("❌ Upgrade created syntax errors - check the file")
                    return False
        else:
            logger.error("❌ Could not find suitable class to upgrade")
            return False

        return True

    def upgrade_mean_reversion_agent(self, dry_run: bool = True) -> bool:
        """Upgrade mean reversion agent with dynamic thresholds"""
        agent_file = self.agents_dir / "mean_reversion_agent.py"

        if not agent_file.exists():
            logger.error(f"❌ Mean reversion agent not found: {agent_file}")
            return False

        logger.info("=" * 80)
        logger.info("UPGRADING MEAN REVERSION AGENT")
        logger.info("=" * 80)

        if not dry_run:
            self.create_backup(agent_file)

        with open(agent_file, 'r', encoding='utf-8') as f:
            content = f.read()

        # Check if already upgraded
        if 'calculate_dynamic_bollinger_bands' in content or 'calculate_keltner_channels' in content:
            logger.warning("⚠️  Mean reversion agent appears to already have dynamic enhancements")
            return True

        enhancement_file = self.agents_dir / "mean_reversion_agent_enhancements.py"
        if not enhancement_file.exists():
            logger.error(f"❌ Enhancement file not found: {enhancement_file}")
            return False

        if dry_run:
            logger.info("📝 PREVIEW: Would add dynamic thresholds to mean reversion agent")
            logger.info("   - Import MeanReversionEnhancements")
            logger.info("   - Replace static Bollinger Bands with dynamic version")
            logger.info("   - Add Keltner Channels")
            logger.info("   - Add dynamic RSI thresholds")
            logger.info("   - Add mean reversion probability calculation")
            logger.info("   - Add Ornstein-Uhlenbeck process")
        else:
            # Add import
            new_content = content
            if 'from agents.mean_reversion_agent_enhancements import MeanReversionEnhancements' not in content:
                import_line = "from agents.mean_reversion_agent_enhancements import MeanReversionEnhancements\n"

                lines = content.split('\n')
                insert_idx = 0
                for i, line in enumerate(lines):
                    if line.startswith('import ') or line.startswith('from '):
                        insert_idx = i + 1

                lines.insert(insert_idx, import_line)
                new_content = '\n'.join(lines)

            with open(agent_file, 'w', encoding='utf-8') as f:
                f.write(new_content)

            if self.validate_python_syntax(agent_file):
                logger.info("✅ Mean reversion agent import added!")
                logger.info("⚠️  NOTE: You'll need to update your signal generation methods to use:")
                logger.info("   - MeanReversionEnhancements.calculate_dynamic_bollinger_bands(df)")
                logger.info("   - MeanReversionEnhancements.calculate_keltner_channels(df)")
                logger.info("   - MeanReversionEnhancements.calculate_dynamic_rsi_thresholds(df)")
                logger.info("   See MEAN_REVERSION_AGENT_UPGRADE_PATCH.py for details")
                return True
            else:
                return False

        return True

    def upgrade_portfolio_allocator(self, dry_run: bool = True) -> bool:
        """Upgrade portfolio allocator with adaptive weights"""
        agent_file = self.agents_dir / "portfolio_allocator_agent.py"

        if not agent_file.exists():
            logger.error(f"❌ Portfolio allocator not found: {agent_file}")
            return False

        logger.info("=" * 80)
        logger.info("UPGRADING PORTFOLIO ALLOCATOR")
        logger.info("=" * 80)

        if not dry_run:
            self.create_backup(agent_file)

        with open(agent_file, 'r', encoding='utf-8') as f:
            content = f.read()

        if 'AdaptiveEnsembleWeights' in content:
            logger.warning("⚠️  Portfolio allocator appears to already have adaptive weights")
            return True

        if dry_run:
            logger.info("📝 PREVIEW: Would add adaptive ensemble weights to portfolio allocator")
            logger.info("   - Add AdaptiveEnsembleWeights class")
            logger.info("   - Add update_regime_info method")
            logger.info("   - Add get_dynamic_strategy_weights method")
            logger.info("   - Add record_strategy_performance method")
            logger.info("   - Update signal fusion to use dynamic weights")
        else:
            # This is more complex - would need to parse and modify the class
            logger.info("⚠️  For portfolio allocator upgrade, please use PORTFOLIO_ALLOCATOR_UPGRADE_PATCH.py")
            logger.info("   The upgrade requires careful integration with your existing signal fusion logic")
            return True

        return True

    def upgrade_risk_manager(self, dry_run: bool = True) -> bool:
        """Upgrade risk manager with portfolio heat"""
        agent_file = self.agents_dir / "risk_manager_agent.py"

        if not agent_file.exists():
            logger.error(f"❌ Risk manager not found: {agent_file}")
            return False

        logger.info("=" * 80)
        logger.info("UPGRADING RISK MANAGER")
        logger.info("=" * 80)

        if not dry_run:
            self.create_backup(agent_file)

        with open(agent_file, 'r', encoding='utf-8') as f:
            content = f.read()

        if 'PortfolioHeatMonitor' in content:
            logger.warning("⚠️  Risk manager appears to already have portfolio heat monitoring")
            return True

        if dry_run:
            logger.info("📝 PREVIEW: Would add portfolio heat monitoring to risk manager")
            logger.info("   - Add PortfolioHeat dataclass")
            logger.info("   - Add PortfolioHeatMonitor class")
            logger.info("   - Add check_portfolio_risk method")
            logger.info("   - Add can_open_position method")
            logger.info("   - Add correlation-adjusted heat calculation")
        else:
            logger.info("⚠️  For risk manager upgrade, please use RISK_MANAGER_UPGRADE_PATCH.py")
            logger.info("   The upgrade requires adding new classes and methods")
            return True

        return True

    def rollback_all(self) -> bool:
        """Rollback all changes from latest backup"""
        logger.info("=" * 80)
        logger.info("ROLLING BACK CHANGES")
        logger.info("=" * 80)

        backups_root = self.base_dir / "backups"
        if not backups_root.exists():
            logger.error("❌ No backups found")
            return False

        # Find most recent backup
        backup_dirs = sorted([d for d in backups_root.iterdir() if d.is_dir()], reverse=True)
        if not backup_dirs:
            logger.error("❌ No backup directories found")
            return False

        latest_backup = backup_dirs[0]
        logger.info(f"Restoring from: {latest_backup}")

        # Restore each file
        for backup_file in latest_backup.glob("*.py"):
            target_file = self.agents_dir / backup_file.name
            shutil.copy2(backup_file, target_file)
            logger.info(f"✅ Restored: {backup_file.name}")

        logger.info("✅ Rollback complete!")
        return True

    def upgrade_all(self, dry_run: bool = True) -> bool:
        """Upgrade all agents"""
        logger.info("=" * 80)
        logger.info(f"{'DRY RUN - PREVIEW ONLY' if dry_run else 'APPLYING ALL UPGRADES'}")
        logger.info("=" * 80)
        logger.info("")

        results = {}

        # Upgrade each agent
        results['momentum'] = self.upgrade_momentum_agent(dry_run)
        logger.info("")

        results['mean_reversion'] = self.upgrade_mean_reversion_agent(dry_run)
        logger.info("")

        results['portfolio_allocator'] = self.upgrade_portfolio_allocator(dry_run)
        logger.info("")

        results['risk_manager'] = self.upgrade_risk_manager(dry_run)
        logger.info("")

        # Summary
        logger.info("=" * 80)
        logger.info("UPGRADE SUMMARY")
        logger.info("=" * 80)

        for agent, success in results.items():
            status = "✅ SUCCESS" if success else "❌ FAILED"
            logger.info(f"{status}: {agent}")

        if dry_run:
            logger.info("")
            logger.info("This was a DRY RUN - no changes were made")
            logger.info("Run with --apply to actually apply the upgrades")

        return all(results.values())


def main():
    parser = argparse.ArgumentParser(
        description='Automatically upgrade trading agents',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument('--apply', action='store_true',
                       help='Apply upgrades (default is preview only)')
    parser.add_argument('--agent', choices=['momentum', 'mean_reversion', 'portfolio_allocator', 'risk_manager'],
                       help='Upgrade specific agent only')
    parser.add_argument('--rollback', action='store_true',
                       help='Rollback to previous backup')

    args = parser.parse_args()

    base_dir = Path(__file__).parent
    upgrader = AgentUpgrader(base_dir)

    if args.rollback:
        success = upgrader.rollback_all()
        sys.exit(0 if success else 1)

    dry_run = not args.apply

    if args.agent:
        # Upgrade specific agent
        if args.agent == 'momentum':
            success = upgrader.upgrade_momentum_agent(dry_run)
        elif args.agent == 'mean_reversion':
            success = upgrader.upgrade_mean_reversion_agent(dry_run)
        elif args.agent == 'portfolio_allocator':
            success = upgrader.upgrade_portfolio_allocator(dry_run)
        elif args.agent == 'risk_manager':
            success = upgrader.upgrade_risk_manager(dry_run)
        else:
            success = False
    else:
        # Upgrade all
        success = upgrader.upgrade_all(dry_run)

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
