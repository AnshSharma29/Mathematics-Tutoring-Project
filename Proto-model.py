import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score
import random
import math
import matplotlib.pyplot as plt


class MathTutoringSystem:
    def __init__(self, min_range=1, max_range=10, learning_rate=0.1):
        self.min_range = min_range
        self.max_range = max_range
        self.current_score = 0
        self.attempts = 0
        self.consecutive_correct_answers = 0
        self.learning_rate = learning_rate
        self.difficulty_levels = {
            "Easy": (1, 10),
            "Medium": (1, 20),
            "Hard": (1, 30),
            "Challenging": (1, 40),
            "Expert": (1, 50)
        }
        self.operations = ["Addition", "Subtraction", "Multiplication", "Division", "Trigonometry"]
        self.selected_operation = None


        # Operation-wise count
        self.operation_counts = {op: 0 for op in self.operations}


        # Lists to store data for plotting accuracy graph
        self.accuracy_history = []
        self.difficulty_levels_history = []

        # Initialize the Random Forest model
        self.model = RandomForestClassifier(random_state=42)

    def train_model(self, X, y):
        self.model.fit(X, y)

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
            print(f"Difficulty increased to {next_difficulty} level! Learning Rate: {self.learning_rate}")
            self.difficulty_levels_history.append(next_difficulty)


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
        if self.selected_operation == "Trigonometry":
            angle = random.uniform(0, 2 * math.pi)
            return angle, None
        else:
            num1 = random.randint(self.min_range, self.max_range)
            num2 = random.randint(self.min_range, self.max_range)
            return num1, num2

    def ask_question(self):
        problem = self.generate_problem()

        if self.selected_operation == "Trigonometry":
            user_answer = float(input(f"What is the sin({problem[0]:.2f})? "))
            correct_answer = round(math.sin(problem[0]), 2)
        else:
            num1, num2 = problem
            if self.selected_operation == "Addition":
                answer = num1 + num2
            elif self.selected_operation == "Subtraction":
                answer = num1 - num2
            elif self.selected_operation == "Multiplication":
                answer = num1 * num2
            elif self.selected_operation == "Division":
                answer = num1 / num2

            user_answer = float(input(f"What is {num1} {self.selected_operation.lower()} {num2}? "))
            correct_answer = answer

        return user_answer == correct_answer, correct_answer

    def provide_explanation(self, correct_answer):
        print(f"Sorry, that's incorrect. The correct answer is {correct_answer}.")
        print("Let me explain that:")
        if self.selected_operation == "Addition" or self.selected_operation == "Subtraction":
            print("When you add or subtract two numbers, you simply combine or subtract them.")
        elif self.selected_operation == "Multiplication":
            print("When you multiply two numbers, you find the total of adding one number to itself as many times as the other number.")
        elif self.selected_operation == "Division":
            print("When you divide two numbers, you find how many times one number fits into the other.")
        elif self.selected_operation == "Trigonometry":
            print("The sine of an angle in a right-angled triangle is the ratio of the length of the side opposite the angle to the length of the hypotenuse.")

        print(f"In this case, you had to {self.selected_operation.lower()} the given numbers.")
        print("Make sure to double-check your calculation.")

    def evaluate_model(self, X_test, y_test):
        y_pred = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')

        print(f"Model Accuracy: {accuracy:.2%}")
        print(f"Model Precision: {precision:.2%}")
        print(f"Model Recall: {recall:.2%}")
        self.accuracy_history.append(accuracy)

    def run_tutoring_session(self, X_train, y_train, X_test, y_test):
        print("Welcome to the Math Tutoring System!")

        # Train the Random Forest model
        self.train_model(X_train, y_train)

        while True:
            self.select_operation()
            while True:
                self.adjust_difficulty()
                correct, answer = self.ask_question()
                self.attempts += 1
                self.operation_counts[self.selected_operation] += 1

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
                    self.evaluate_model(X_test, y_test)

                    # Print the operation-wise count
                    print("\nOperation-wise Questions:")
                    for op, count in self.operation_counts.items():
                        print(f"{op}: {count}")

                    # Plot the accuracy graph
                    self.plot_accuracy_graph()

                    exit_choice = input("Do you want to exit the tutoring session? (yes/no) ").lower()
                    if exit_choice == 'yes':
                        print("Thanks for using the Math Tutoring System. Goodbye!")
                        return  # Exit the tutoring session
                    else:
                        break  # Go back to the operation selection

    def plot_accuracy_graph(self):
        plt.figure(figsize=(8, 5))
        plt.plot(self.accuracy_history, marker='o', linestyle='-', color='b')
        plt.title('Accuracy Over Time')
        plt.xlabel('Session')
        plt.ylabel('Accuracy')
        plt.grid(True)
        plt.show()


if __name__ == "__main__":
    # Generate synthetic data for training and testing the model
    np.random.seed(42)
    X = np.random.rand(100, 2)
    y = (X[:, 0] + X[:, 1] > 1).astype(int)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    tutor = MathTutoringSystem()
    tutor.run_tutoring_session(X_train, y_train, X_test, y_test)


