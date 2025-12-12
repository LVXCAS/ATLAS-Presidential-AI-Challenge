"""Session Timing Agent - Optimal trading session detection."""
from typing import Dict, Tuple
from .base_agent import BaseAgent
from datetime import datetime

class SessionTimingAgent(BaseAgent):
    def __init__(self, initial_weight: float = 1.2):
        super().__init__(name="SessionTimingAgent", initial_weight=initial_weight)

    def analyze(self, market_data: Dict) -> Tuple[str, float, Dict]:
        hour = datetime.utcnow().hour
        
        # London (8-12 UTC) and NY (13-17 UTC) sessions
        if 8 <= hour < 12 or 13 <= hour < 17:
            session = "optimal"
            confidence = 0.65
        else:
            session = "suboptimal"
            confidence = 0.3
            
        return ("NEUTRAL", confidence, {"session": session, "hour": hour})
