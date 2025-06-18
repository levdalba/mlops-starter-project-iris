import io
import os
import time

import chess.pgn
import mlflow
import pandas as pd

# Import for custom metrics
from mlflow.metrics.genai import EvaluationExample, answer_similarity

# Load OpenAI API key from environment variable (set this in your shell or .env file)
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("Please set the OPENAI_API_KEY environment variable.")

# Explicitly set the tracking URI to match the running MLflow server
mlflow.set_tracking_uri("http://0.0.0.0:5001")

# Enable OpenAI autologging for tracing
mlflow.openai.autolog()

# PGN data for Magnus Carlsen's game
pgn = io.StringIO(
    """[Event "Troll Masters"]
[Site "Gausdal NOR"]
[Date "2001.01.05"]
[Round "1"]
[White "Edvardsen,R"]
[Black "Carlsen,Magnus"]
[Result "1/2-1/2"]
[WhiteElo "2055"]
[BlackElo ""]
[ECO "D12"]

1.d4 Nf6 2.Nf3 d5 3.e3 Bf5 4.c4 c6 5.Nc3 e6 6.Bd3 Bxd3 7.Qxd3 Nbd7 8.b3 Bd6
9.O-O O-O 10.Bb2 Qe7 11.Rad1 Rad8 12.Rfe1 dxc4 13.bxc4 e5 14.dxe5 Nxe5 15.Nxe5 Bxe5
16.Qe2 Rxd1 17.Rxd1 Rd8 18.Rxd8+ Qxd8 19.Qd1 Qxd1+ 20.Nxd1 Bxb2 21.Nxb2 b5
22.f3 Kf8 23.Kf2 Ke7 1/2-1/2"""
)

# Parse PGN to get FENs and moves for Magnus's turns
game = chess.pgn.read_game(pgn)
if game is None:
    print("Failed to parse PGN game")
    exit(1)
print("Game parsed successfully")

board = game.board()
moves = []
fens = []
for node in game.mainline():
    if board.turn == chess.BLACK and node.move:
        fens.append(board.fen())
        moves.append(node.move)
    board.push(node.move)

print("Moves:", [move.uci() for move in moves])
print("FENs:", fens)

# Generate a unique prompt template based on run time
run_timestamp = int(time.time())
prompt_variations = [
    (
        "Given a chess position in FEN format (e.g., 'rnbqkbnr/pppppppp/8/8/3P4/8/"
        "PPP1PPPP/RNBQKBNR b KQkq - 0 1'), predict the move Magnus Carlsen would "
        "most likely play based on a fine-tuned Lc0 model trained on his 6,000+ "
        "games, focusing on positional control. (Run {run_timestamp})"
    ),
    (
        "Given a chess position in FEN format, predict Magnus Carlsen's move using "
        "a tactical approach with a fine-tuned Lc0 model trained on his 6,000+ "
        "games. (Run {run_timestamp})"
    ),
    (
        "Analyze a chess position in FEN format and predict Magnus Carlsen's most "
        "probable move, emphasizing endgame strategy, using a fine-tuned Lc0 model "
        "trained on his 6,000+ games. (Run {run_timestamp})"
    ),
    (
        "Evaluate a chess position in FEN format and suggest Magnus Carlsen's "
        "likely move, prioritizing defensive maneuvers, with a fine-tuned Lc0 "
        "model trained on his 6,000+ games. (Run {run_timestamp})"
    ),
]
prompt_index = run_timestamp % len(prompt_variations)
unique_prompt_template = prompt_variations[prompt_index].format(
    run_timestamp=run_timestamp
)

# Register prompt within the run
with mlflow.start_run() as run:
    system_prompt = mlflow.genai.register_prompt(
        name="magnus_chess_prompt",
        template=unique_prompt_template,
        commit_message=("Unique prompt for Magnus Carlsen move prediction"),
    )
    print(f"Prompt object: {system_prompt}")
    print(f"Available attributes: {dir(system_prompt)}")
    print(f"Prompt version: {system_prompt.version}")
    mlflow.log_param("prompt_name", system_prompt.name)
    mlflow.log_param("run_timestamp", run_timestamp)
    mlflow.log_text(system_prompt.template, "prompt_template.txt")

# Simulate evaluation DataFrame with placeholder predictions
eval_df = pd.DataFrame(
    {
        "inputs": fens[:5],  # First 5 positions
        "ground_truth": [move.uci() for move in moves[:5]],  # True moves
        "predictions": [
            "g8f6",
            "b8c6",
            "c7c6",
            "e7e6",
            "b8c6",
        ],  # Placeholder predictions
    }
)

# Custom answer similarity metric (simulated) with OpenAI
example = EvaluationExample(
    input=fens[0],
    output=moves[0].uci(),
    score=5,
    justification="The predicted move exactly matches the true move played by Magnus.",
    grading_context={"targets": moves[0].uci()},
)

answer_similarity_metric = answer_similarity(model="openai:/gpt-4", examples=[example])

# Evaluate with MLflow and manually log custom metrics
with mlflow.start_run() as run:
    eval_dataset = mlflow.data.from_pandas(
        df=eval_df,
        name="magnus_move_eval_simulated",
        targets="ground_truth",
        predictions="predictions",
    )
    mlflow.log_input(eval_dataset)
    result = mlflow.evaluate(
        data=eval_dataset,
        extra_metrics=[answer_similarity_metric],
        evaluator_config={"col_mapping": {"inputs": "inputs"}},
    )

    # Manually log custom metrics
    accuracy_scores = [
        1.0 if pred == true else 0.0
        for pred, true in zip(
            eval_df["predictions"], eval_df["ground_truth"], strict=False
        )
    ]
    top3_moves = [m.uci() for m in moves[:3]]
    top3_accuracy_scores = [
        1.0 if pred in top3_moves else 0.0 for pred in eval_df["predictions"]
    ]

    for i, (acc, top3_acc) in enumerate(
        zip(accuracy_scores, top3_accuracy_scores, strict=False)
    ):
        mlflow.log_metric(f"accuracy_{i}", acc)
        mlflow.log_metric(f"top3_accuracy_{i}", top3_acc)

    # Log average metrics for summary
    mlflow.log_metric("accuracy_mean", sum(accuracy_scores) / len(accuracy_scores))
    mlflow.log_metric(
        "top3_accuracy_mean", sum(top3_accuracy_scores) / len(top3_accuracy_scores)
    )

# Display results for screenshot
print(result.metrics)
print(result.tables["eval_results_table"])
