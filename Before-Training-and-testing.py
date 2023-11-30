import random

class MathTutoringSystem:
    def __init__(self, min_range=1, max_range=10):
        self.min_range = min_range
        self.max_range = max_range
        self.current_score = 0
        self.attempts = 0
        self.consecutive_correct_answers = 0
        self.difficulty_levels = {
            "Easy": (1, 10),
            "Medium": (1, 20),
            "Hard": (1, 30),
            "Challenging": (1, 40),
            "Expert": (1, 50)
        }
        self.operations = ["Addition", "Subtraction", "Multiplication", "Division"]
        self.selected_operation = None

    def get_difficulty_level(self):
        for level, (min_val, max_val) in self.difficulty_levels.items():
            if self.min_range == min_val and self.max_range == max_val:
                return level

    def adjust_difficulty(self):
        # Adjust difficulty based on performance
        if self.consecutive_correct_answers == 3:
            # Increase difficulty after three consecutive correct answers
            self.increase_difficulty()
            self.consecutive_correct_answers = 0
        elif self.current_score >= 5:
            # Gradual increase in difficulty based on overall performance
            self.increase_difficulty()

    def increase_difficulty(self):
        current_difficulty = self.get_difficulty_level()
        difficulty_keys = list(self.difficulty_levels.keys())
        current_index = difficulty_keys.index(current_difficulty)
        if current_index < len(difficulty_keys) - 1:
            # Increase difficulty level
            next_difficulty = difficulty_keys[current_index + 1]
            self.min_range, self.max_range = self.difficulty_levels[next_difficulty]
            print(f"Difficulty increased to {next_difficulty} level!")

    def select_operation(self):
        print("Choose the operation:")
        for i, operation in enumerate(self.operations, start=1):
            print(f"{i}. {operation}")
        choice = int(input("Enter the number corresponding to your choice: "))
        if 1 <= choice <= len(self.operations):
            self.selected_operation = self.operations[choice - 1]
        else:
            print("Invalid choice. Defaulting to Addition.")
            self.selected_operation = "Addition"

    def generate_problem(self):
        num1 = random.randint(self.min_range, self.max_range)
        num2 = random.randint(self.min_range, self.max_range)
        return num1, num2

    def ask_question(self):
        num1, num2 = self.generate_problem()

        if self.selected_operation == "Addition":
            answer = num1 + num2
            user_answer = int(input(f"What is {num1} + {num2}? "))
        elif self.selected_operation == "Subtraction":
            answer = num1 - num2
            user_answer = int(input(f"What is {num1} - {num2}? "))
        elif self.selected_operation == "Multiplication":
            answer = num1 * num2
            user_answer = int(input(f"What is {num1} * {num2}? "))
        elif self.selected_operation == "Division":
            answer = num1 / num2
            user_answer = float(input(f"What is {num1} / {num2}? "))

        return user_answer == answer, answer

    def provide_explanation(self, correct_answer):
        print(f"Sorry, that's incorrect. The correct answer is {correct_answer}.")
        print("Let me explain that:")
        if self.selected_operation == "Addition" or self.selected_operation == "Subtraction":
            print("When you add or subtract two numbers, you simply combine or subtract them.")
        elif self.selected_operation == "Multiplication":
            print("When you multiply two numbers, you find the total of adding one number to itself as many times as the other number.")
        elif self.selected_operation == "Division":
            print("When you divide two numbers, you find how many times one number fits into the other.")

        print(f"In this case, you had to {self.selected_operation.lower()} the given numbers.")
        print("Make sure to double-check your calculation.")

    def run_tutoring_session(self):
        print("Welcome to the Math Tutoring System!")

        while True:
            self.select_operation()
            while True:
                self.adjust_difficulty()
                correct, answer = self.ask_question()
                self.attempts += 1

                if correct:
                    self.consecutive_correct_answers += 1
                    self.current_score += 1
                    print("Correct! Well done.")
                else:
                    self.consecutive_correct_answers = 0
                    print("Incorrect. Let's try another one.")
                    self.provide_explanation(answer)

                print(f"Current Score: {self.current_score}/{self.attempts}")
                print(f"Current Difficulty: {self.get_difficulty_level()}")

                next_question = input("Do you want another question? (yes/no) ").lower()

                if next_question != 'yes':
                    exit_choice = input("Do you want to exit the tutoring session? (yes/no) ").lower()
                    if exit_choice == 'yes':
                        print("Thanks for using the Math Tutoring System. Goodbye!")
                        return  # Exit the tutoring session
                    else:
                        break  # Go back to the operation selection


if __name__ == "__main__":
    tutor = MathTutoringSystem()
    tutor.run_tutoring_session()
