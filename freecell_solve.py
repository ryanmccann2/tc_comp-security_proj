import heapq
import copy
import itertools
from tqdm import tqdm


table = [['QS', 'AS', '5D', '7C', '3S', '4H', '2H'],
        ['7H', '5S', '8C', '6D', '10H', '2S', '9H'],
        ['8S', 'AD', '2C', 'KD', '5C', '8D', '10S'],
        ['3D', '9S', '7S', '5H', 'KS', '3H', '9D'],
        ['KC', 'QH', 'QD', '10D', 'QC', '7D'],
        ['10C', 'JD', 'JH', '6S', '6H', 'AC'],
        ['JS', '6C', 'AH', '2D', '4S', '8H'],
        ['JC', '9C', '3C', 'KH', '4D', '4C'],
]
test_tableau_easy = [
    ['KC', 'QC', 'JC', '10C', '9C', '8C', '7C', '6C', '5C', '4C', '3C', '2C', 'AC'],  # All clubs in descending order
    ['KD', 'QD', 'JD', '10D', '9D', '8D', '7D', '6D', '5D', '4D', '3D', '2D', 'AD'],  # All diamonds in descending order
    ['KH', 'QH', 'JH', '10H', '9H', '8H', '7H', '6H', '5H', '4H', '3H', '2H', 'AH'],  # All hearts in descending order
    ['KS', 'QS', 'JS', '10S', '9S', '8S', '7S', '6S', '5S', '4S', '3S', '2S', 'AS'],  # All spades in descending order
    [],  # Empty column
    [],  # Empty column
    [],  # Empty column
    []   # Empty column
]

visited = set()

class FreeCellGame:
    def __init__(self, tableau):
        self.foundations = {'H': 0, 'D': 0, 'C': 0, 'S': 0}
        self.free_cells = [0] * 4               # 4 free cells
        self.tableau = copy.deepcopy(tableau)   # deep copy to preserve original game state
        self.history = []                       # track moves for backtracking and solution
        self.visited_states = set()
        self.cards_moved = set()

    def __lt__(self, a):
        if self.heuristic == a.heuristic():
            return self.len(history) < a.len(history)
        else:
            return self.heuristic() < a.heuristic()

    def __str__(self):
        return f"Tableau: {self.tableau}\nFree Cells: {self.free_cells}\nFoundations: {self.foundations}\nHistory: {self.history}\nHeuristic: {self.heuristic()}\nHash: {self.state_identifier()}\n"

    def generate_possible_moves(self):
        moves = []
        # Check each column in the tableau for possible moves
        for i, column in enumerate(self.tableau):
            if column:
                top_card = column[-1]
                # Move to other tableau columns
                for j, target_column in enumerate(self.tableau):
                    if i != j and self.can_move_to_tableau(top_card, target_column):
                        moves.append(('pile_to_pile', i, j, top_card))
                                # Move to foundation
                if self.can_move_to_foundation(top_card):
                    moves.append(('pile_to_foundation', i, None, top_card))
                # Move to free cells
                if self.can_move_to_free_cell(top_card):
                    moves.append(('pile_to_free', i, None, top_card))



        # Check free cells for possible moves back to tableau or foundations
        for i, card in enumerate(self.free_cells):
            if card != 0:
                # Move from free cell to tableau
                for j, column in enumerate(self.tableau):
                    if self.can_move_to_tableau(card, column):
                        moves.append(('free_to_pile', i, j, card))
                # Move from free cell to foundation
                if self.can_move_to_foundation(card):
                    moves.append(('free_to_foundation', i, None, card))

        return moves

    def card_map(self, card):
        card_val_dict = {'A': 1, '2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8, '9': 9, '10': 10, 'J': 11, 'Q': 12, 'K': 13}
        return card_val_dict[card[:-1]]

    def can_move_to_free_cell(self, card):
        return any(cell == 0 for cell in self.free_cells)

    def move_to_free_cell(self, card):
        for i, cell in enumerate(self.free_cells):
            if cell == 0:
                self.free_cells[i] = card
                break

    def can_move_to_foundation(self, card):
        if self.card_map(card) - 1 == self.foundations[card[-1]]:
            return True
        return False

    def move_to_foundation(self, card):
        if self.can_move_to_foundation(card):
            self.foundations[card[-1]] += 1

    def can_move_to_tableau(self, card, dest):
        if not dest:
            return True
        if (self.card_red(card) and self.card_black(dest[-1])) or (self.card_black(card) and self.card_red(dest[-1])):
            if self.card_map(card) + 1 == self.card_map(dest[-1]):
                return True
        return False

    def move_to_tableau(self, card, dest):
        if can_move_to_tableau(card, dest):
            dest.append(card)

    def card_red(self, card):
        return card[-1] in 'HD'

    def card_black(self, card):
        return card[-1] in 'SC'

    def apply_move(self, move):
        type, src, tgt, card = move
        self.cards_moved.add(card)
        new_game = copy.deepcopy(self)
        if type == 'pile_to_free':
            new_game.tableau[src].pop()
            new_game.move_to_free_cell(card)
        elif type == 'pile_to_foundation':
            new_game.tableau[src].pop()
            new_game.move_to_foundation(card)
        elif type == 'free_to_tableau':
            new_game.free_cells[src] = 0
            new_game.tableau[tgt].append(card)
        elif type == 'free_to_foundation':
            new_game.free_cells[src] = 0
            new_game.move_to_foundation(card)
        elif type == 'pile_to_pile':
            new_game.tableau[src].pop()
            new_game.tableau[tgt].append(card)
        new_game.history.append(move)
        return new_game

    def heuristic(self):
        # Adjust the heuristic function to prioritize states with more cards in the foundations:
        return sum(self.foundations.values()) * 5 + len(self.cards_moved) * 2 - len([0 for card in self.free_cells if card != 0]) # Sum of ranks in foundations

    def is_winner(self):
        # kings in all foundation piles
        return all(value == 13 for value in self.foundations.values())

    def state_identifier(self):
        # Create a unique identifier for the state for detecting revisits
        return hash(tuple(tuple(row) for row in self.tableau) + tuple(self.foundations.items()))

def hints(initial_game, d=10):
    # Priority queue for states, using negative heuristic scores for max-heap behavior
    top_5 = []
    queue = [(-initial_game.heuristic(), 0, initial_game)]
    heapq.heappush(top_5, (-initial_game.heuristic(), 0, initial_game))

    pbar = tqdm(total = d)
    while queue:
        current_score, depth, current_state = queue.pop(0)
        pbar.update(depth - pbar.n)
        if current_state.is_winner():
            print("Winning state found!")
            return [current_state]

        if depth < d:  # Only explore up to depth of d moves: default is 5
            for move in current_state.generate_possible_moves():
                new_state = current_state.apply_move(move) # apply move

                state_id = new_state.state_identifier()
                if state_id not in visited:
                    visited.add(state_id) # visit state
                    heapq.heappush(top_5, (-new_state.heuristic(), depth + 1, new_state)) # add state to queue
                    queue.append((-new_state.heuristic(), depth + 1, new_state))

    top_states = heapq.nsmallest(5, top_5)
    print(top_states)
    return [state for _, _, state in sorted(top_states, reverse=True)]


def print_move(move):
    tag, to, fro, card = move
    if tag == 'pile_to_free':
        print(f"Move column {to}: {card} to free cell")
    elif tag == 'pile_to_foundation':
        print(f"Move column {to}: {card} to foundation")
    elif tag == 'free_to_pile':
        print(f"Move free cell {to}: {card} to column {fro}")
    elif tag == 'free_to_foundation':
        print(f"Move free cell {to}: {card} to foundation")
    elif tag == 'pile_to_pile':
        print(f"Move column {to}: {card} to column {fro}")
    print(f"{tag} + {to}  {fro}, card")

# game = FreeCellGame(table)
game = FreeCellGame(test_tableau_easy)
states = [game]
generation_number = 0

last_states = 0
while not states[0].is_winner() and generation_number < 1000 and last_states != sum([state.state_identifier() for state in states]):

    print(f'generation_number: {generation_number} last_gen_hash: {last_states} {hash(str(states))}')
    last_states = sum([state.state_identifier() for state in states])
    potential_solutions = []
    for state in states:
        potential_solutions += hints(state)

    heapq.heapify(potential_solutions)
    states = sorted(heapq.nsmallest(5, potential_solutions))

    flag = False
    for state in states:
        print(state)
        if state.is_winner():
            flag = True

    if flag:
        break

    input('Wait')
    generation_number += 1

# random number for now
if states[-1].heuristic() < 200:
    print('Unsolvable')
else:
    print("Winner!")