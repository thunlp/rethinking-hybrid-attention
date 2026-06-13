import json
import random
import re
from pathlib import Path
from tqdm import tqdm
from argparse import ArgumentParser
from common import (
    build_dataset_name,
    build_dataset_spec,
    build_depths,
    DIFFICULTY_CHOICES,
    get_default_data_dir,
)

try:
    import wonderwords
except ImportError:
    wonderwords = None
    print("Warning: wonderwords is not installed. Using the built-in key word list.")

DEFAULT_DATA_DIR = get_default_data_dir(__file__)
NLTK_CANDIDATES = [
    DEFAULT_DATA_DIR / "data" / "nltk_data",
    DEFAULT_DATA_DIR / "nltk_data",
]
USE_NLTK = False
try:
    import nltk
    from nltk.tokenize import sent_tokenize
    custom_nltk_path = next((p for p in NLTK_CANDIDATES if p.exists()), None)
    if custom_nltk_path is not None:
        custom_nltk_path_str = str(custom_nltk_path)
        if custom_nltk_path_str not in nltk.data.path:
            nltk.data.path.insert(0, custom_nltk_path_str)
        print(f"Loaded custom NLTK path: {custom_nltk_path_str}")
    
    try:
        nltk.data.find('tokenizers/punkt')
        test_text = "This is a test. This is another test!"
        sent_tokenize(test_text)
        USE_NLTK = True
        print("NLTK punkt tokenizer is available.")
        
    except LookupError:
        print("Warning: tokenizers/punkt was not found in the configured nltk_data directory.")
        if custom_nltk_path is not None:
            print(f"Expected path: {custom_nltk_path}/tokenizers/punkt/")
        USE_NLTK = False
    except Exception as e:
        print(f"Warning: failed to load NLTK punkt tokenizer: {e}")
        USE_NLTK = False

except ImportError:
    print("Warning: NLTK is not installed. Using the regex sentence tokenizer.")
    USE_NLTK = False

# Fallback sentence tokenizer (simple regex-based)
def simple_sent_tokenize(text):
    """Simple sentence tokenizer using regex as fallback"""
    # Split on sentence endings (. ! ?) followed by space or newline
    sentences = re.split(r'([.!?]+)\s+', text)
    # Recombine sentences with their punctuation
    result = []
    for i in range(0, len(sentences) - 1, 2):
        if i + 1 < len(sentences):
            result.append(sentences[i] + sentences[i + 1])
        else:
            result.append(sentences[i])
    if len(sentences) % 2 == 1:
        result.append(sentences[-1])
    # Filter out empty strings
    return [s.strip() for s in result if s.strip()]

# Use NLTK if available, otherwise use fallback
if USE_NLTK:
    sent_tokenize_func = sent_tokenize
else:
    sent_tokenize_func = simple_sent_tokenize

parser = ArgumentParser()
parser.add_argument("--save_dir", type=str, default=str(DEFAULT_DATA_DIR), help="Directory to save data")
parser.add_argument("--num_samples", type=int, default=2000, help="Number of samples to generate")
parser.add_argument("--seq_len", type=int, default=16000, help="Target sequence length")
parser.add_argument("--seed", type=int, default=42, help="Random seed")
parser.add_argument("--haystack_type", type=str, default="repeat", choices=["repeat", "essay"], 
                    help="Type of haystack: 'repeat' (niah_single_1 style) or 'essay' (niah_single_2 style)")
parser.add_argument("--essay_path", type=str, default=None, 
                    help="Path to PaulGrahamEssays.json (default: auto-detect)")
parser.add_argument("--num_classes", type=int, default=4, help="Number of candidate magic numbers/classes to sample from")
parser.add_argument(
    "--difficulty",
    type=str,
    default="easy",
    choices=DIFFICULTY_CHOICES,
    help="Difficulty tier for candidate magic numbers.",
)
parser.add_argument("--depth_min", type=int, default=0, help="Minimum essay insertion depth percentage")
parser.add_argument("--depth_max", type=int, default=99, help="Maximum essay insertion depth percentage")
parser.add_argument("--depth_step", type=int, default=3, help="Step size for essay insertion depth percentages")
args = parser.parse_args()

if args.num_classes < 2:
    raise ValueError("--num_classes must be at least 2")
if not (0 <= args.depth_min <= 100 and 0 <= args.depth_max <= 100):
    raise ValueError("--depth_min and --depth_max must be between 0 and 100")
if args.depth_min > args.depth_max:
    raise ValueError("--depth_min must be <= --depth_max")
if args.depth_step <= 0:
    raise ValueError("--depth_step must be > 0")

# Initialize random seed
random.seed(args.seed)

# Load words for generating keys (adj-noun format)
words = None
try:
    if wonderwords is not None:
        nouns = wonderwords.random_word._get_words_from_text_file("nounlist.txt")
        adjs = wonderwords.random_word._get_words_from_text_file("adjectivelist.txt")
        words = [f"{adj}-{noun}" for adj in adjs for noun in nouns]
        words = sorted(list(set(words)))
    else:
        raise RuntimeError("wonderwords unavailable")
except Exception as e:
    # Fallback: use a deterministic built-in word list if wonderwords is unavailable
    builtin_adjs = [
        "amber", "brisk", "calm", "crisp", "eager", "fancy", "gentle", "golden",
        "grand", "green", "hidden", "icy", "jolly", "lucky", "misty", "modern",
        "nimble", "novel", "olive", "polar", "quiet", "rapid", "royal", "rustic",
        "silver", "solar", "steady", "sunny", "swift", "tidy", "urban", "vivid",
    ]
    builtin_nouns = [
        "anchor", "breeze", "cedar", "comet", "crystal", "delta", "ember", "falcon",
        "forest", "galaxy", "harbor", "island", "lantern", "meadow", "meteor", "oasis",
        "orchid", "paddle", "pebble", "quartz", "radar", "river", "shadow", "signal",
        "spruce", "summit", "thunder", "valley", "voyage", "willow", "zephyr", "beacon",
    ]
    words = [f"{adj}-{noun}" for adj in builtin_adjs for noun in builtin_nouns]
    print(f"Warning: Using built-in fallback word list. Error: {e}")

if words is None or len(words) == 0:
    words = [f"word-{i}" for i in range(1000)]

# Haystack: repeated sentences (as in niah_single_1)
HAYSTACK_REPEAT = "The grass is green. The sky is blue. The sun is yellow. Here we go. There and back again."

# Template (single form since num_needle_q * num_needle_v == 1)
TEMPLATE = """A special magic number is hidden within the following text. Make sure to memorize it. I will quiz you about the number afterwards.
{context}
What is the special magic number for {query} mentioned in the provided text? The special magic number for {query} mentioned in the provided text is"""

# Candidate magic numbers for configurable classification.
MAGIC_NUMBERS_EASY = [
    "1234567",
    "2345678",
    "3456789",
    "4567890",
    "5678901",
    "6789012",
    "7890123",
    "8901234",
]


MAGIC_NUMBERS_HARD = [
    "31415920",
    "31415921",
    "31415922",
    "31415923",
    "31415924",
    "31415925",
    "31415926",
    "31415927",
]

MAGIC_POOL_BY_DIFFICULTY = {
    "easy": MAGIC_NUMBERS_EASY,
    "hard": MAGIC_NUMBERS_HARD,
}
MAGIC_NUMBERS = MAGIC_POOL_BY_DIFFICULTY[args.difficulty]

if args.num_classes > len(MAGIC_NUMBERS):
    raise ValueError(
        f"--num_classes={args.num_classes} exceeds available candidate numbers ({len(MAGIC_NUMBERS)}) "
        f"for difficulty={args.difficulty}"
    )

# Positions for essay insertion (similar to niah.py)
DEPTHS = build_depths(args.depth_min, args.depth_max, args.depth_step)

def generate_magic_number(num_classes):
    label = random.randrange(num_classes)
    number_str = MAGIC_NUMBERS[label]
    return number_str, label

def insert_needle_by_depth(document_sents, needle):
    depth = random.choice(DEPTHS)
    insertion_pos = int(len(document_sents) * (depth / 100))
    insertion_pos = min(insertion_pos, len(document_sents))
    document_sents_list = list(document_sents)
    document_sents_list.insert(insertion_pos, needle)
    return " ".join(document_sents_list), depth

def generate_random_word():
    return random.choice(words)

def load_essay_haystack(essay_path=None):
    if essay_path is None:
        possible_paths = [
            # Preferred local probing data locations.
            DEFAULT_DATA_DIR / "data" / "PaulGrahamEssays.json",
            DEFAULT_DATA_DIR / "PaulGrahamEssays.json",
            # Legacy in-repo location.
            Path(__file__).parent / "json" / "PaulGrahamEssays.json",
        ]
        for path in possible_paths:
            if path.exists():
                essay_path = path
                break
        
        if essay_path is None or not Path(essay_path).exists():
            searched = "\n".join(str(p) for p in possible_paths)
            raise FileNotFoundError(
                "PaulGrahamEssays.json not found. Searched paths:\n" + searched
            )
    
    essay_path = Path(essay_path)
    print(f"Using essay haystack file: {essay_path}")
    with open(essay_path, 'r', encoding='utf-8') as f:
        essay_data = json.load(f)
    
    essay_text = essay_data.get('text', '')
    haystack_words = re.sub(r'\s+', " ", essay_text).split(" ")
    return haystack_words

def generate_input_output_repeat(num_haystack_repeats):
    key = generate_random_word()
    value_str, label = generate_magic_number(args.num_classes)
    needle = f"One of the special magic numbers for {key} is: {value_str}."
    sentences = [HAYSTACK_REPEAT] * num_haystack_repeats
    insert_pos = random.randint(len(sentences) // 4, len(sentences)*3 // 4)
    sentences.insert(insert_pos, needle)
    context = "\n".join(sentences)
    input_text = TEMPLATE.format(context=context, query=key)
    return input_text, label, key, value_str

def generate_input_output_essay(haystack_words, num_haystack_words):
    key = generate_random_word()
    value_str, label = generate_magic_number(args.num_classes)
    needle = f"One of the special magic numbers for {key} is: {value_str}."
    start_idx = random.randint(0, max(0, len(haystack_words) - num_haystack_words))
    text = " ".join(haystack_words[start_idx:start_idx + num_haystack_words])
    
    document_sents = sent_tokenize_func(text.strip())
    
    if len(document_sents) == 0:
        context = text + " " + needle
        depth = None
    else:
        context, depth = insert_needle_by_depth(document_sents, needle)
    
    input_text = TEMPLATE.format(context=context, query=key)
    return input_text, label, key, value_str, depth

def main():
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    spec = build_dataset_spec(
        haystack_type=args.haystack_type,
        num_samples=args.num_samples,
        seq_len=args.seq_len,
        num_classes=args.num_classes,
        difficulty=args.difficulty,
        depth_min=args.depth_min,
        depth_max=args.depth_max,
        depth_step=args.depth_step,
    )
    dataset_name = build_dataset_name(spec)
    save_path = save_dir / f"probing_data_{dataset_name}.json"
    print(f"Using difficulty={args.difficulty} with {args.num_classes} classes.")
    
    dataset = []
    
    if args.haystack_type == "essay":
        print(f"Loading essay haystack...")
        haystack_words = load_essay_haystack(args.essay_path)
        num_haystack_words = max(1000, args.seq_len // 4)
        num_haystack_words = min(num_haystack_words, len(haystack_words))
        print(f"Using depth candidates: {DEPTHS}")
    else:
        num_haystack_repeats = max(100, args.seq_len // 100)
    
    for i in tqdm(range(args.num_samples), desc="Generating samples"):
        if args.haystack_type == "essay":
            input_text, label, key, value_str, depth = generate_input_output_essay(haystack_words, num_haystack_words)
        else:
            input_text, label, key, value_str = generate_input_output_repeat(num_haystack_repeats)
            depth = None
        
        dataset.append({
            "id": i,
            "text": input_text,
            "label": label,
            "value": value_str,
            "key": key,
            "haystack_type": args.haystack_type,
            "num_classes": args.num_classes,
            "difficulty": args.difficulty,
            "depth": depth,
            "depth_candidates": DEPTHS if args.haystack_type == "essay" else None,
        })
    
    with open(save_path, "w") as f:
        json.dump(dataset, f, indent=2)
    
    print(f"\nData generation completed. Saved to: {save_path}")

if __name__ == "__main__":
    main()