import sys
from config import Config

class SemanticValidator:
    def __init__(self):
        self.provider = Config.LLM_PROVIDER # 'openai', 'ollama', or 'terminal'
        # For this urgent request, we default to 'terminal' if not specified in env
        # In a real app, logic would be cleaner
        pass

    def check_causality(self, stock_ticker: str, macro_name: str) -> tuple:
        """
        Asks source: Does Macro X drive Stock Y?
        Returns: (is_valid: bool, reasoning: str)
        """
        # 1. Terminal / Manual Mode
        # Useful for debugging or free usage where user acts as the Brain.
        if self.provider == 'terminal' or self.provider == 'manual':
            print(f"\n🧠 [Logic Check Required]")
            print(f"❓ Does '{macro_name}' logically drive '{stock_ticker}'?")
            print("   (Economic Causality Check)")
            
            # Interactive Input
            # Since we might be running in a non-interactive agent environment, 
            # we need to be careful. If no input possible, auto-approve for testing?
            # But user ASKED for this.
            
            try:
                user_input = input("   👉 Type 'y' for YES, 'n' for NO (Reason optional, separate with comma): ")
            except EOFError:
                # If input is not possible (e.g. background process), default to safe False
                return False, "Input stream closed (Non-interactive mode)"
                
            is_valid = user_input.lower().startswith('y')
            reason = "Human Validation"
            if "," in user_input:
                parts = user_input.split(",", 1)
                reason = parts[1].strip()
                
            return is_valid, reason

        # 2. Local LLM / OpenAI (Future Sprint)
        # Placeholder for external API call
        return True, "Auto-approved (Mock)"

if __name__ == "__main__":
    # Test
    val = SemanticValidator()
    # Force provider for test
    val.provider = 'terminal' 
    valid, reason = val.check_causality("1605.TW", "Copper Prices")
    print(f"Result: {valid} | Reason: {reason}")
