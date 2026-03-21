"""
Daifugo GUI - Pygame version
Play Human vs AI graphically
"""

import pygame
import sys
import time
from daifugo_env import Daifugo1v1Env, Meld, PASS, State, rank_of, rank_value, RANKS
from wrapper import DaifugoGymEnv
from train import DaifugoTrainer
import numpy as np
from dataclasses import dataclass


@dataclass
class CardRect:
    """Card rectangle information"""
    card_id: str
    rect: pygame.Rect
    action_id: int = -1


class DaifugoGUI:
    """Daifugo GUI"""
    
    # Color definitions
    BG_COLOR = (34, 139, 34)  # Green
    TEXT_COLOR = (255, 255, 255)  # White
    CARD_COLOR = (245, 245, 245)  # White
    CARD_BORDER = (0, 0, 0)  # Black
    HIGHLIGHT_COLOR = (255, 215, 0)  # Gold
    
    # Card display settings
    CARD_WIDTH = 80
    CARD_HEIGHT = 120
    CARD_SPACING = 10
    
    def __init__(self, screen_width: int = 1200, screen_height: int = 800, model=None):
        pygame.init()
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.screen = pygame.display.set_mode((screen_width, screen_height))
        pygame.display.set_caption("Daifugo - Human vs AI")
        self.clock = pygame.time.Clock()
        
        # Font settings - use default fonts (ASCII compatible)
        self.font_large = pygame.font.Font(None, 36)
        self.font_medium = pygame.font.Font(None, 24)
        self.font_small = pygame.font.Font(None, 18)
        self.font_tiny = pygame.font.Font(None, 14)
        
        # Game state
        self.env = DaifugoGymEnv(seed=int(time.time() * 1000000) % (2**31 - 1))
        self.env.player = 0  # Human is player 0
        self.obs, self.info = self.env.reset()
        self.model = model
        self.selected_card_idx = None
        self.message = "Game Start!"
        self.message_timer = 0
        
        # Scroll functionality
        self.hand_scroll_offset = 0
        self.max_scroll_offset = 0
        self.hand_display_width = self.screen_width - 40  # 20px margin on each side
        
        # Multi-card selection
        self.selected_cards = []  # List of selected card indices
        self.keys_down = set()  # Track held keys
    
    def get_card_text(self, card_id: str) -> str:
        """Convert card ID to display text"""
        if card_id == "JR":
            return "JR\nRED"
        elif card_id == "JB":
            return "JB\nBLK"
        else:
            suit_map = {"S": "S", "H": "H", "D": "D", "C": "C"}
            suit = suit_map.get(card_id[0], card_id[0])
            rank = card_id[1:]
            return f"{suit}\n{rank}"
    
    def get_hand_cards(self) -> list[CardRect]:
        """Get player's hand cards as list (with cache and scroll support)"""
        if not hasattr(self, '_hand_cards_cache'):
            hand = list(self.env.state.hands[self.env.player])
            
            # Sort by strength
            def card_strength(card: str) -> tuple:
                rank = rank_of(card)
                value = rank_value(rank, self.env.state.revolution)
                return (-value, card[0] if len(card) > 1 else "", rank)
            
            hand_sorted = sorted(hand, key=card_strength)
            
            # Calculate rectangles
            total_width = len(hand_sorted) * (self.CARD_WIDTH + self.CARD_SPACING)
            self.max_scroll_offset = max(0, total_width - self.hand_display_width)
            
            start_y = self.screen_height - self.CARD_HEIGHT - 40
            
            cards = []
            for i, card in enumerate(hand_sorted):
                x = 20 + i * (self.CARD_WIDTH + self.CARD_SPACING) - self.hand_scroll_offset
                rect = pygame.Rect(x, start_y, self.CARD_WIDTH, self.CARD_HEIGHT)
                
                # Get action ID
                legal_actions = self.env.env.legal_actions(self.env.state)
                legal_ids = [
                    self.env.env.action_index[self.env.env._canon(a)] for a in legal_actions
                ]
                
                # Find single card action for this card
                action_id = -1
                for idx in legal_ids:
                    action = self.env.env.action_table[idx]
                    if action != PASS and action.meld_type == "single" and card in action.cards:
                        action_id = idx
                        break
                
                cards.append(CardRect(card, rect, action_id))
            
            self._hand_cards_cache = cards
        
        return self._hand_cards_cache
    
    def clear_hand_cache(self):
        """Clear hand card cache"""
        if hasattr(self, '_hand_cards_cache'):
            del self._hand_cards_cache
    
    def draw_card(self, x: int, y: int, card_id: str = None, is_selected: bool = False, is_back: bool = False):
        """Draw a card"""
        rect = pygame.Rect(x, y, self.CARD_WIDTH, self.CARD_HEIGHT)
        
        # Card background
        color = self.HIGHLIGHT_COLOR if is_selected else self.CARD_COLOR
        pygame.draw.rect(self.screen, color, rect)
        pygame.draw.rect(self.screen, self.CARD_BORDER, rect, 2)
        
        # Card back or front
        if is_back:
            # Card back (AI hand)
            pygame.draw.rect(self.screen, (100, 100, 100), rect.inflate(-4, -4))
            text = self.font_small.render("?", True, self.TEXT_COLOR)
            text_rect = text.get_rect(center=rect.center)
            self.screen.blit(text, text_rect)
        elif card_id:
            # Card front (player hand)
            lines = self.get_card_text(card_id).split("\n")
            for i, line in enumerate(lines):
                text = self.font_medium.render(line, True, (0, 0, 0))
                text_rect = text.get_rect(center=(rect.centerx, rect.centery - 10 + i * 30))
                self.screen.blit(text, text_rect)
    
    def draw_table(self):
        """Draw table state (centered)"""
        y = 380  # Centered vertically on screen
        
        if self.env.state.table is None:
            text = "Table: Clear (Start new play)"
            color = (200, 200, 200)
        else:
            state = self.env.state
            meld_type_map = {"single": "SINGLE", "group": "GROUP", "straight": "STRAIGHT"}
            meld_type = meld_type_map[state.table.meld_type]
            text = f"Table: {meld_type} ({state.table.size} cards) - Top: {state.table.top_rank}"
            color = self.TEXT_COLOR
        
        # Center the table display
        text_surface = self.font_medium.render(text, True, color)
        text_rect = text_surface.get_rect(center=(self.screen_width // 2, y))
        self.screen.blit(text_surface, text_rect)
    
    def draw_hand_info(self):
        """Draw hand information"""
        y = 250
        self.screen.blit(self.font_medium.render("Hand Info:", True, self.TEXT_COLOR), (20, y))
        
        my_cards = len(self.env.state.hands[0])
        ai_cards = len(self.env.state.hands[1])
        
        color_me = (0, 255, 0) if self.env.state.turn == 0 else (200, 200, 200)
        color_ai = (255, 100, 100) if self.env.state.turn == 1 else (200, 200, 200)
        
        self.screen.blit(self.font_small.render(f"You: {my_cards} cards", True, color_me), (20, y + 30))
        self.screen.blit(self.font_small.render(f"AI: {ai_cards} cards (secret)", True, color_ai), (20, y + 60))
    
    def draw_turn_info(self):
        """Draw turn information"""
        y = 350
        current_turn = "Your Turn" if self.env.state.turn == 0 else "AI Turn"
        color = (0, 255, 0) if self.env.state.turn == 0 else (255, 100, 100)
        
        self.screen.blit(self.font_large.render(current_turn, True, color), (20, y))
        
        if self.env.state.revolution:
            self.screen.blit(self.font_medium.render("REVOLUTION!", True, (255, 215, 0)), (20, y + 40))
    
    def draw_legal_actions(self):
        """Draw legal actions"""
        if self.env.state.turn != 0:
            return  # Don't show during AI turn
        
        legal_actions = self.env.env.legal_actions(self.env.state)
        legal_ids = [self.env.env.action_index[self.env.env._canon(a)] for a in legal_actions]
        
        x = self.screen_width - 300
        y = 100
        
        self.screen.blit(self.font_medium.render("Legal Moves:", True, self.TEXT_COLOR), (x, y))
        
        count = 0
        for action_id in legal_ids[:5]:  # Show first 5 only
            action = self.env.env.action_table[action_id]
            if action == PASS:
                text = "PASS"
            else:
                meld_type_map = {"single": "S", "group": "G", "straight": "St"}
                meld_type = meld_type_map[action.meld_type]
                text = f"[{meld_type}] ({action.size} cards)"
            
            self.screen.blit(self.font_small.render(text, True, self.TEXT_COLOR), (x, y + 40 + count * 25))
            count += 1
        
        if len(legal_ids) > 5:
            self.screen.blit(self.font_small.render(f"... +{len(legal_ids) - 5} more", True, (200, 200, 200)), (x, y + 40 + count * 25))
    
    def draw_message(self):
        """Display message"""
        if self.message_timer > 0:
            text = self.font_small.render(self.message, True, (255, 215, 0))
            text_rect = text.get_rect(bottomleft=(20, self.screen_height - 10))
            self.screen.blit(text, text_rect)
            self.message_timer -= 1
    
    def draw_players_hand(self):
        """Draw AI hand (back) and player hand (front)"""
        # AI hand (back)
        ai_hand_count = len(self.env.state.hands[1])
        ai_start_x = 50
        for i in range(min(ai_hand_count, 8)):  # Show max 8 cards
            self.draw_card(ai_start_x + i * 60, 30, is_back=True)
        
        if ai_hand_count > 8:
            text = self.font_small.render(f"... +{ai_hand_count - 8} cards", True, self.TEXT_COLOR)
            self.screen.blit(text, (ai_start_x + 8 * 60 + 10, 30))
        
        # Player hand (front) - with scroll support
        hand_cards = self.get_hand_cards()
        
        # Set clip region (only draw within display area)
        clip_rect = pygame.Rect(20, self.screen_height - self.CARD_HEIGHT - 40, self.hand_display_width, self.CARD_HEIGHT + 20)
        old_clip = self.screen.get_clip()
        self.screen.set_clip(clip_rect)
        
        for i, card in enumerate(hand_cards):
            is_selected = i in self.selected_cards
            self.draw_card(card.rect.x, card.rect.y, card.card_id, is_selected=is_selected)
        
        self.screen.set_clip(old_clip)
        
        # Scroll indicator
        if self.max_scroll_offset > 0:
            # Left arrow
            if self.hand_scroll_offset > 0:
                text = self.font_small.render("<", True, self.HIGHLIGHT_COLOR)
                self.screen.blit(text, (10, self.screen_height - self.CARD_HEIGHT - 20))
            
            # Right arrow
            if self.hand_scroll_offset < self.max_scroll_offset:
                text = self.font_small.render(">", True, self.HIGHLIGHT_COLOR)
                self.screen.blit(text, (self.screen_width - 30, self.screen_height - self.CARD_HEIGHT - 20))
            
            # Scroll info
            scroll_text = self.font_tiny.render("Use LEFT/RIGHT arrow keys to scroll", True, (200, 200, 200))
            self.screen.blit(scroll_text, (self.screen_width - 250, self.screen_height - 25))
    
    def find_action_for_cards(self, card_indices: list) -> int:
        """Find action ID that matches all selected cards"""
        if not card_indices:
            return -1
        
        hand_cards = self.get_hand_cards()
        selected_card_ids = [hand_cards[i].card_id for i in card_indices]
        selected_set = set(selected_card_ids)
        
        legal_actions = self.env.env.legal_actions(self.env.state)
        legal_ids = [
            self.env.env.action_index[self.env.env._canon(a)] for a in legal_actions
        ]
        
        # Find action that matches selected cards
        for idx in legal_ids:
            action = self.env.env.action_table[idx]
            if action != PASS:
                action_cards_set = set(action.cards)
                if action_cards_set == selected_set:
                    return idx
        
        return -1
    
    def handle_click(self, pos):
        """Handle mouse click - support multi-card selection"""
        if self.env.state.turn != 0:
            self.message = "AI is playing..."
            self.message_timer = 120
            return
        
        hand_cards = self.get_hand_cards()
        
        for i, card in enumerate(hand_cards):
            if card.rect.collidepoint(pos):
                # Toggle card selection (no need to check action_id for clicking)
                if i in self.selected_cards:
                    self.selected_cards.remove(i)
                    self.message = "Card deselected"
                else:
                    self.selected_cards.append(i)
                    self.message = f"Card selected ({len(self.selected_cards)} selected)"
                self.message_timer = 60
                break
    
    def handle_key(self, key):
        """Handle keyboard input"""
        if key == pygame.K_SPACE and self.selected_cards and self.env.state.turn == 0:
            # Space key to confirm - find action matching all selected cards
            action_id = self.find_action_for_cards(self.selected_cards)
            if action_id != -1:
                self.execute_action(action_id)
            else:
                self.message = "Invalid card combination"
                self.message_timer = 120
        elif key == pygame.K_p and self.env.state.turn == 0:
            # P key for PASS (only during player's turn) - always allowed
            try:
                pass_action_id = self.env.env.action_index[self.env.env._canon(PASS)]
                self.execute_action(pass_action_id)
            except (KeyError, AttributeError):
                self.message = "PASS action not available"
                self.message_timer = 120
        elif key == pygame.K_ESCAPE:
            # ESC key to deselect all
            self.selected_cards = []
            self.message = "Selection cleared"
            self.message_timer = 30
        elif key == pygame.K_r and self.env.state.done:
            # R key to restart (only when game is over)
            self.restart_game()
        elif key == pygame.K_LEFT:
            self.keys_down.add(pygame.K_LEFT)
        elif key == pygame.K_RIGHT:
            self.keys_down.add(pygame.K_RIGHT)
    
    def handle_key_up(self, key):
        """Handle keyboard release"""
        self.keys_down.discard(key)
    
    def update_scroll(self):
        """Update scroll based on held keys (called every frame)"""
        if pygame.K_LEFT in self.keys_down:
            self.hand_scroll_offset = max(0, self.hand_scroll_offset - 25)
            self.clear_hand_cache()
        if pygame.K_RIGHT in self.keys_down:
            self.hand_scroll_offset = min(self.max_scroll_offset, self.hand_scroll_offset + 25)
            self.clear_hand_cache()
    
    def restart_game(self):
        """Restart the game"""
        # Create new environment with time-based seed
        self.env = DaifugoGymEnv(seed=int(time.time() * 1000000) % (2**31 - 1))
        self.env.player = 0
        self.obs, self.info = self.env.reset()
        self.selected_cards = []
        self.hand_scroll_offset = 0
        self.message = "Game restarted!"
        self.message_timer = 120
        self.clear_hand_cache()
    
    def execute_action(self, action_id: int):
        """Execute action"""
        try:
            self.obs, reward, done, _, self.info = self.env.step(action_id)
            self.clear_hand_cache()
            self.selected_cards = []  # Clear all selections after playing
            
            action = self.env.env.action_table[action_id]
            if action == PASS:
                self.message = "PASS"
            else:
                self.message = f"Played {action.size} card(s)"
            self.message_timer = 120
            
            # AI turn
            if not done and self.env.state.turn != 0:
                pygame.display.flip()
                pygame.time.wait(500)  # 0.5 second delay
                self.ai_turn()
            
            if done:
                winner = "You" if self.env.state.last_winner == 0 else "AI"
                self.message = f"Game Over! Winner: {winner}"
                self.message_timer = 300
        except ValueError as e:
            self.message = "Invalid move"
            self.message_timer = 120
    
    def ai_turn(self):
        """Execute AI turn"""
        turn_count = 0
        while self.env.state.turn != 0 and not self.env.state.done and turn_count < 100:
            turn_count += 1
            
            try:
                if self.model is not None:
                    action_id, _ = self.model.predict(self.obs, deterministic=True)
                    action_id = int(action_id)
                else:
                    mask = self.info["action_mask"]
                    legal_ids = np.where(mask)[0]
                    action_id = np.random.choice(legal_ids)
                
                # Execute action
                self.obs, _, _, _, self.info = self.env.step(action_id)
                self.clear_hand_cache()
            except Exception as e:
                self.message = f"AI Error: {str(e)[:20]}"
                self.message_timer = 120
                break
        
        pygame.display.flip()
        pygame.time.wait(500)
    
    def draw(self):
        """Draw screen"""
        self.screen.fill(self.BG_COLOR)
        
        # Game info
        self.draw_table()
        self.draw_hand_info()
        self.draw_turn_info()
        self.draw_legal_actions()
        
        # Play area
        self.draw_players_hand()
        
        # Message
        self.draw_message()
        
        # Game over screen
        if self.env.state.done:
            self.draw_game_over_screen()
        
        # Controls guide
        if self.env.state.turn == 0 and not self.env.state.done:
            guide = "Click to select - SPACE to play / P for PASS / ESC to deselect / LEFT/RIGHT to scroll"
            self.screen.blit(self.font_small.render(guide, True, (200, 200, 200)), (20, self.screen_height - 30))
        
        pygame.display.flip()
    
    def draw_game_over_screen(self):
        """Draw game over screen overlay"""
        # Semi-transparent overlay
        overlay = pygame.Surface((self.screen_width, self.screen_height))
        overlay.set_alpha(200)
        overlay.fill((0, 0, 0))
        self.screen.blit(overlay, (0, 0))
        
        # Game over message
        winner = "You WIN!" if self.env.state.last_winner == 0 else "AI WINS!"
        color = (0, 255, 0) if self.env.state.last_winner == 0 else (255, 100, 100)
        
        text = self.font_large.render("GAME OVER", True, (255, 215, 0))
        text_rect = text.get_rect(center=(self.screen_width // 2, self.screen_height // 2 - 60))
        self.screen.blit(text, text_rect)
        
        text2 = self.font_large.render(winner, True, color)
        text_rect2 = text2.get_rect(center=(self.screen_width // 2, self.screen_height // 2))
        self.screen.blit(text2, text_rect2)
        
        # Restart instruction
        text3 = self.font_medium.render("Press R to restart", True, (200, 200, 200))
        text_rect3 = text3.get_rect(center=(self.screen_width // 2, self.screen_height // 2 + 60))
        self.screen.blit(text3, text_rect3)
    
    def run(self):
        """Main loop"""
        running = True
        
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    self.handle_click(event.pos)
                elif event.type == pygame.KEYDOWN:
                    self.handle_key(event.key)
                elif event.type == pygame.KEYUP:
                    self.handle_key_up(event.key)
            
            self.update_scroll()  # Update scroll from held keys
            self.draw()
            self.clock.tick(60)  # 60 FPS
        
        pygame.quit()
        sys.exit()


def main():
    print("Starting Daifugo GUI...\n")
    
    # Load model
    print("Loading trained model...")
    try:
        trainer = DaifugoTrainer()
        trainer.load_model("models/daifugo_ppo.zip")
        model = trainer.model
        print("OK - Model loaded!\n")
    except FileNotFoundError:
        print("WARNING - Model not found. Playing with random AI.\n")
        model = None
    
    # Launch GUI
    gui = DaifugoGUI(model=model)
    gui.run()


if __name__ == "__main__":
    main()
