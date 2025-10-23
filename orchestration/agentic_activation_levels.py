#!/usr/bin/env python3
"""
AGENTIC ACTIVATION SYSTEM
Build full agentic AI, activate features gradually

Level 0: Pure rules (training wheels ON)
Level 1: AI pattern recognition
Level 2: AI entry timing
Level 3: AI position sizing
Level 4: AI strategy selection
Level 5: AI exit management
Level 6: AI risk management
Level 7: AI portfolio optimization
Level 8: AI crisis response
Level 9: AI self-optimization
Level 10: FULL AUTONOMY (no human oversight)

Start at Level 0 Monday.
Increase 1 level per week as you validate.
Reach Level 10 by Week 10.
"""

from enum import Enum
from dataclasses import dataclass
from typing import Dict, List, Optional
import json
from datetime import datetime


class AgenticLevel(Enum):
    """10 levels of AI autonomy"""
    LEVEL_0_PURE_RULES = 0           # Human controls everything
    LEVEL_1_AI_PATTERNS = 1          # AI finds patterns, human decides
    LEVEL_2_AI_TIMING = 2            # AI picks entry times
    LEVEL_3_AI_SIZING = 3            # AI sizes positions
    LEVEL_4_AI_STRATEGY = 4          # AI selects strategies
    LEVEL_5_AI_EXITS = 5             # AI manages exits
    LEVEL_6_AI_RISK = 6              # AI manages risk
    LEVEL_7_AI_PORTFOLIO = 7         # AI optimizes portfolio
    LEVEL_8_AI_CRISIS = 8            # AI handles crises
    LEVEL_9_AI_SELF_OPTIMIZE = 9     # AI improves itself
    LEVEL_10_FULL_AUTONOMY = 10      # Complete AI control


@dataclass
class AgenticConfig:
    """Configuration for agentic system"""
    current_level: AgenticLevel
    features_enabled: List[str]
    safety_overrides: Dict[str, any]
    performance_metrics: Dict[str, float]
    activation_date: str


class AgenticActivationSystem:
    """
    Manages gradual activation of AI features

    Each level adds more AI autonomy while maintaining safety
    """

    def __init__(self, starting_level: AgenticLevel = AgenticLevel.LEVEL_0_PURE_RULES):
        self.current_level = starting_level
        self.config = self._load_or_create_config()

    def _load_or_create_config(self) -> AgenticConfig:
        """Load existing config or create new"""
        try:
            with open('agentic_config.json', 'r') as f:
                data = json.load(f)
                return AgenticConfig(**data)
        except FileNotFoundError:
            # Create default config
            return AgenticConfig(
                current_level=self.current_level,
                features_enabled=[],
                safety_overrides={
                    'max_loss_per_trade': 0.02,  # 2% max
                    'max_positions': 10,
                    'require_human_approval': True,
                    'stop_on_drawdown': 0.10,  # Stop if down 10%
                },
                performance_metrics={
                    'trades_executed': 0,
                    'win_rate': 0.0,
                    'avg_profit': 0.0,
                    'max_drawdown': 0.0,
                },
                activation_date=datetime.now().isoformat()
            )

    def get_capabilities(self, level: AgenticLevel) -> Dict:
        """Get what AI can do at each level"""

        capabilities = {
            AgenticLevel.LEVEL_0_PURE_RULES: {
                'description': 'Training Wheels ON - Pure rule-based',
                'ai_controls': [],
                'human_controls': [
                    'Strategy selection',
                    'Position sizing',
                    'Entry timing',
                    'Exit timing',
                    'Risk management',
                    'Portfolio allocation'
                ],
                'safety': 'Maximum - Human approves everything',
                'recommended_duration': '1-2 weeks (20-30 trades)',
                'activation_criteria': 'None - Start here Monday'
            },

            AgenticLevel.LEVEL_1_AI_PATTERNS: {
                'description': 'AI Pattern Recognition Active',
                'ai_controls': [
                    'Pattern detection (ML models)',
                    'Opportunity scoring',
                    'Technical indicator analysis'
                ],
                'human_controls': [
                    'Strategy selection',
                    'Position sizing',
                    'Entry/Exit timing',
                    'Risk management'
                ],
                'safety': 'High - AI only provides recommendations',
                'recommended_duration': '1 week (10-20 trades)',
                'activation_criteria': 'Win rate >60% at Level 0'
            },

            AgenticLevel.LEVEL_2_AI_TIMING: {
                'description': 'AI Entry Timing Optimization',
                'ai_controls': [
                    'Pattern detection',
                    'Entry timing (when to enter)',
                    'Optimal strike selection'
                ],
                'human_controls': [
                    'Strategy selection',
                    'Position sizing',
                    'Exit timing',
                    'Risk limits'
                ],
                'safety': 'High - Human sets boundaries',
                'recommended_duration': '1 week',
                'activation_criteria': 'Win rate >65% at Level 1'
            },

            AgenticLevel.LEVEL_3_AI_SIZING: {
                'description': 'AI Position Sizing',
                'ai_controls': [
                    'Pattern detection',
                    'Entry timing',
                    'Position sizing (within limits)',
                    'Kelly Criterion optimization'
                ],
                'human_controls': [
                    'Strategy selection',
                    'Exit timing',
                    'Maximum risk per trade'
                ],
                'safety': 'Medium - AI sizes but within hard limits',
                'recommended_duration': '1 week',
                'activation_criteria': 'Win rate >68% at Level 2'
            },

            AgenticLevel.LEVEL_4_AI_STRATEGY: {
                'description': 'AI Strategy Selection',
                'ai_controls': [
                    'Pattern detection',
                    'Entry timing',
                    'Position sizing',
                    'Strategy selection (from approved list)'
                ],
                'human_controls': [
                    'Approved strategy list',
                    'Exit timing',
                    'Risk limits'
                ],
                'safety': 'Medium - AI picks from safe strategies only',
                'recommended_duration': '1 week',
                'activation_criteria': 'Win rate >70% at Level 3'
            },

            AgenticLevel.LEVEL_5_AI_EXITS: {
                'description': 'AI Exit Management',
                'ai_controls': [
                    'Pattern detection',
                    'Entry/Exit timing',
                    'Position sizing',
                    'Strategy selection',
                    'Stop loss optimization'
                ],
                'human_controls': [
                    'Maximum loss limits',
                    'Portfolio allocation'
                ],
                'safety': 'Medium - Stop losses still enforced',
                'recommended_duration': '1-2 weeks',
                'activation_criteria': 'Win rate >70%, Max DD <15%'
            },

            AgenticLevel.LEVEL_6_AI_RISK: {
                'description': 'AI Risk Management',
                'ai_controls': [
                    'All previous',
                    'Dynamic risk adjustment',
                    'Correlation analysis',
                    'Volatility-based sizing'
                ],
                'human_controls': [
                    'Maximum portfolio risk',
                    'Crisis override'
                ],
                'safety': 'Medium-Low - AI adjusts risk dynamically',
                'recommended_duration': '2 weeks',
                'activation_criteria': 'Win rate >72%, consistent profits'
            },

            AgenticLevel.LEVEL_7_AI_PORTFOLIO: {
                'description': 'AI Portfolio Optimization',
                'ai_controls': [
                    'All previous',
                    'Multi-strategy balancing',
                    'Portfolio rebalancing',
                    'Capital allocation'
                ],
                'human_controls': [
                    'Overall portfolio limits',
                    'Strategy whitelist'
                ],
                'safety': 'Low - AI manages entire portfolio',
                'recommended_duration': '2 weeks',
                'activation_criteria': 'Win rate >75%, Sharpe >2.0'
            },

            AgenticLevel.LEVEL_8_AI_CRISIS: {
                'description': 'AI Crisis Response',
                'ai_controls': [
                    'All previous',
                    'Automatic crisis detection',
                    'Emergency hedging',
                    'Volatility trading'
                ],
                'human_controls': [
                    'Maximum emergency actions',
                    'Human notification required'
                ],
                'safety': 'Low - AI acts in emergencies',
                'recommended_duration': '2-3 weeks',
                'activation_criteria': 'Survived 1+ crisis successfully'
            },

            AgenticLevel.LEVEL_9_AI_SELF_OPTIMIZE: {
                'description': 'AI Self-Optimization',
                'ai_controls': [
                    'All previous',
                    'Strategy parameter tuning',
                    'Model retraining',
                    'Performance optimization'
                ],
                'human_controls': [
                    'Approve optimizations',
                    'Performance bounds'
                ],
                'safety': 'Very Low - AI improves itself',
                'recommended_duration': '3-4 weeks',
                'activation_criteria': 'Win rate >80%, 6+ months track record'
            },

            AgenticLevel.LEVEL_10_FULL_AUTONOMY: {
                'description': 'FULL AI AUTONOMY',
                'ai_controls': [
                    'EVERYTHING - Complete control',
                    'No human approval needed',
                    'Self-optimization',
                    'Strategy creation'
                ],
                'human_controls': [
                    'Emergency stop button only'
                ],
                'safety': 'MINIMAL - AI fully autonomous',
                'recommended_duration': 'Ongoing',
                'activation_criteria': 'Win rate >85%, 12+ months proven, PhD team oversight'
            }
        }

        return capabilities.get(level, {})

    def should_level_up(self) -> tuple[bool, str]:
        """Check if ready to increase autonomy level"""

        current_caps = self.get_capabilities(self.current_level)
        criteria = current_caps.get('activation_criteria', '')

        # Get current metrics
        metrics = self.config.performance_metrics
        trades = metrics.get('trades_executed', 0)
        win_rate = metrics.get('win_rate', 0)
        drawdown = metrics.get('max_drawdown', 0)

        # Level-specific checks
        if self.current_level == AgenticLevel.LEVEL_0_PURE_RULES:
            if trades >= 20 and win_rate >= 0.60:
                return (True, f"Ready for Level 1: {trades} trades, {win_rate:.1%} win rate")
            return (False, f"Need 20+ trades at 60%+ win rate (current: {trades} trades, {win_rate:.1%})")

        elif self.current_level == AgenticLevel.LEVEL_1_AI_PATTERNS:
            if trades >= 30 and win_rate >= 0.65:
                return (True, f"Ready for Level 2: {win_rate:.1%} win rate validated")
            return (False, f"Need 30+ trades at 65%+ win rate")

        elif self.current_level == AgenticLevel.LEVEL_2_AI_TIMING:
            if trades >= 40 and win_rate >= 0.68:
                return (True, f"Ready for Level 3: Entry timing optimized")
            return (False, f"Need 40+ trades at 68%+ win rate")

        elif self.current_level == AgenticLevel.LEVEL_3_AI_SIZING:
            if trades >= 50 and win_rate >= 0.70:
                return (True, f"Ready for Level 4: {win_rate:.1%} win rate achieved!")
            return (False, f"Need 50+ trades at 70%+ win rate")

        elif self.current_level == AgenticLevel.LEVEL_4_AI_STRATEGY:
            if trades >= 70 and win_rate >= 0.70 and drawdown < 0.15:
                return (True, f"Ready for Level 5: Strategy selection validated")
            return (False, f"Need 70+ trades, 70%+ win rate, <15% drawdown")

        # Higher levels need more extensive validation
        elif self.current_level == AgenticLevel.LEVEL_5_AI_EXITS:
            if trades >= 100 and win_rate >= 0.72:
                return (True, "Ready for Level 6: Exit management proven")
            return (False, f"Need 100+ trades at 72%+ win rate")

        # Default: need human approval for higher levels
        return (False, f"Manual approval required for {self.current_level.name}")

    def level_up(self, force: bool = False) -> bool:
        """Increase autonomy level"""

        if not force:
            can_level_up, reason = self.should_level_up()
            if not can_level_up:
                print(f"[NOT READY] {reason}")
                return False

        # Increase level
        next_level_value = self.current_level.value + 1
        if next_level_value > 10:
            print("[MAX LEVEL] Already at full autonomy")
            return False

        next_level = AgenticLevel(next_level_value)
        self.current_level = next_level

        print(f"\n{'='*70}")
        print(f"LEVEL UP: {next_level.name}")
        print(f"{'='*70}")

        caps = self.get_capabilities(next_level)
        print(f"\n{caps['description']}")
        print(f"\nAI now controls:")
        for control in caps['ai_controls']:
            print(f"  + {control}")
        print(f"\nHuman still controls:")
        for control in caps['human_controls']:
            print(f"  - {control}")
        print(f"\nSafety level: {caps['safety']}")
        print(f"Duration: {caps['recommended_duration']}")
        print(f"{'='*70}\n")

        # Save config
        self.config.current_level = next_level
        self._save_config()

        return True

    def _save_config(self):
        """Save configuration"""
        with open('agentic_config.json', 'w') as f:
            json.dump({
                'current_level': self.current_level.value,
                'features_enabled': self.config.features_enabled,
                'safety_overrides': self.config.safety_overrides,
                'performance_metrics': self.config.performance_metrics,
                'activation_date': self.config.activation_date
            }, f, indent=2)

    def display_roadmap(self):
        """Show full 10-level roadmap"""

        print("\n" + "="*70)
        print("AGENTIC ACTIVATION ROADMAP - 10 LEVELS")
        print("="*70)

        for level in AgenticLevel:
            caps = self.get_capabilities(level)
            is_current = (level == self.current_level)
            status = " [CURRENT]" if is_current else ""

            print(f"\n{level.name}{status}")
            print(f"  {caps['description']}")
            print(f"  Duration: {caps.get('recommended_duration', 'TBD')}")
            print(f"  Criteria: {caps.get('activation_criteria', 'TBD')}")

        print("\n" + "="*70)
        print(f"Current Level: {self.current_level.name}")
        print(f"Next Level: {AgenticLevel(self.current_level.value + 1).name if self.current_level.value < 10 else 'MAX'}")

        can_level_up, reason = self.should_level_up()
        print(f"Can level up: {'YES' if can_level_up else 'NO'}")
        print(f"Reason: {reason}")
        print("="*70 + "\n")


def main():
    """Demo the agentic activation system"""

    system = AgenticActivationSystem(AgenticLevel.LEVEL_0_PURE_RULES)
    system.display_roadmap()

    print("\n[RECOMMENDATION] Start at Level 0 Monday")
    print("[GOAL] Reach Level 4-5 by Week 6")
    print("[GOAL] Reach Level 10 by Month 3-4")


if __name__ == "__main__":
    main()
