"""
Query processing module for Information Retrieval.
Provides tokenization and parsing for Free-Text, Boolean (AND/OR/NOT), and Phrase queries
using positional postings lists from the InvertedIndex.
"""

import re
import sys
from pathlib import Path
from typing import List, Set, Dict, Optional, Tuple, Any

# Path configuration
sys.path.append(str(Path(__file__).resolve().parent.parent))
from preprocess import tokenize, clean_text


def parse_query_terms(query_str: str) -> List[str]:
    """
    Applies the identical cleaning and tokenization pipeline used during indexing.
    Guarantees consistency between query and corpus terms.
    """
    return tokenize(query_str)


def detect_query_type(query_str: str) -> str:
    """
    Detects query style:
    - 'phrase': wrapped in quotation marks e.g. "call of duty"
    - 'boolean': contains explicit uppercase/lowercase boolean operators (AND, OR, NOT)
    - 'free_text': standard bag-of-words multi-word query
    """
    query_stripped = query_str.strip()
    if query_stripped.startswith('"') and query_stripped.endswith('"') and len(query_stripped) > 2:
        return 'phrase'
    
    # Check for boolean operators as standalone words
    if re.search(r'\b(AND|OR|NOT)\b', query_str, re.IGNORECASE):
        return 'boolean'
        
    return 'free_text'


def evaluate_phrase_query(phrase: str, index: Any) -> List[int]:
    """
    Evaluates a phrase query (e.g. 'black ops') using positional postings.
    Returns doc_ids where the phrase terms appear consecutively in exact order.
    """
    cleaned_phrase = phrase.strip(' "')
    terms = parse_query_terms(cleaned_phrase)
    
    if not terms:
        return []
    
    if len(terms) == 1:
        postings = index.get_postings(terms[0])
        return [doc_id for doc_id, tf, pos in postings]
    
    # Retrieve postings for each term
    term_postings = [index.get_postings(t) for t in terms]
    
    # If any term is not in vocabulary, phrase cannot match
    if any(len(p) == 0 for p in term_postings):
        return []
    
    # Map doc_id -> positions for each term
    doc_pos_maps: List[Dict[int, List[int]]] = []
    for postings in term_postings:
        doc_pos_maps.append({doc_id: pos for doc_id, tf, pos in postings})
        
    # Find candidate documents containing all terms
    candidate_docs = set(doc_pos_maps[0].keys())
    for d_map in doc_pos_maps[1:]:
        candidate_docs &= set(d_map.keys())
        if not candidate_docs:
            return []
            
    matching_docs = []
    for doc_id in sorted(candidate_docs):
        # Check if term1 at pos p implies term2 at p+1, term3 at p+2, etc.
        pos0_list = doc_pos_maps[0][doc_id]
        has_phrase = False
        
        for start_pos in pos0_list:
            matched = True
            for offset, d_map in enumerate(doc_pos_maps[1:], start=1):
                if (start_pos + offset) not in d_map[doc_id]:
                    matched = False
                    break
            if matched:
                has_phrase = True
                break
                
        if has_phrase:
            matching_docs.append(doc_id)
            
    return matching_docs


def evaluate_boolean_query(query_str: str, index: Any) -> Set[int]:
    """
    Evaluates a Boolean query supporting AND, OR, NOT operations with precedence:
    NOT > AND > OR, and parentheses.
    Example: 'xbox AND halo', 'ps5 OR playstation', 'call AND duty NOT cold'
    """
    all_doc_ids = set(range(index.num_docs))
    
    # Tokenize boolean expression
    # Extract words, operators, and parens
    raw_tokens = re.findall(r'\(|\)|\bAND\b|\bOR\b|\bNOT\b|[^\s()]+', query_str, re.IGNORECASE)
    
    # Normalize operators to uppercase, keep terms
    tokens = []
    for tok in raw_tokens:
        tok_upper = tok.upper()
        if tok_upper in ('AND', 'OR', 'NOT', '(', ')'):
            tokens.append(tok_upper)
        else:
            tokens.append(tok)
            
    if not tokens:
        return set()

    # Convert to postfix (Reverse Polish Notation) using Shunting-Yard algorithm
    precedence = {'NOT': 3, 'AND': 2, 'OR': 1}
    output_queue = []
    operator_stack = []
    
    # Insert implicit AND if two terms are adjacent or term followed by NOT
    normalized_tokens = []
    for i, tok in enumerate(tokens):
        normalized_tokens.append(tok)
        if i < len(tokens) - 1:
            next_tok = tokens[i+1]
            if (tok not in ('AND', 'OR', 'NOT', '(') and 
                next_tok not in ('AND', 'OR', ')')):
                normalized_tokens.append('AND')
            elif tok == ')' and next_tok not in ('AND', 'OR', ')'):
                normalized_tokens.append('AND')
                
    for token in normalized_tokens:
        if token not in ('AND', 'OR', 'NOT', '(', ')'):
            output_queue.append(token)
        elif token == '(':
            operator_stack.append(token)
        elif token == ')':
            while operator_stack and operator_stack[-1] != '(':
                output_queue.append(operator_stack.pop())
            if operator_stack and operator_stack[-1] == '(':
                operator_stack.pop()
        else:
            while (operator_stack and operator_stack[-1] != '(' and
                   precedence.get(operator_stack[-1], 0) >= precedence.get(token, 0)):
                output_queue.append(operator_stack.pop())
            operator_stack.append(token)
            
    while operator_stack:
        output_queue.append(operator_stack.pop())
        
    # Evaluate postfix expression
    eval_stack: List[Set[int]] = []
    
    for token in output_queue:
        if token == 'NOT':
            operand = eval_stack.pop() if eval_stack else set()
            eval_stack.append(all_doc_ids - operand)
        elif token == 'AND':
            right = eval_stack.pop() if eval_stack else set()
            left = eval_stack.pop() if eval_stack else set()
            eval_stack.append(left & right)
        elif token == 'OR':
            right = eval_stack.pop() if eval_stack else set()
            left = eval_stack.pop() if eval_stack else set()
            eval_stack.append(left | right)
        else:
            # Clean term and get matching doc_ids from postings
            terms = parse_query_terms(token)
            if terms:
                # If term cleans to multiple words, take intersection
                term_docs = None
                for t in terms:
                    postings = index.get_postings(t)
                    t_doc_ids = {doc_id for doc_id, tf, pos in postings}
                    if term_docs is None:
                        term_docs = t_doc_ids
                    else:
                        term_docs &= t_doc_ids
                eval_stack.append(term_docs if term_docs is not None else set())
            else:
                eval_stack.append(set())
                
    return eval_stack.pop() if eval_stack else set()
