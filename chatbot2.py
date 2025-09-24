import re 
import sys 
import io 
import contextlib 
import traceback 
import ast 
import importlib 
import importlib.util 
from typing import Dict, Any, Tuple, Optional, List, Set, Union 
import requests 
import json
import time 
import subprocess 
import os
from dataclasses import dataclass, field 
from datetime import datetime 
import hashlib
from collections import defaultdict 
from openai import AzureOpenAI 
from dotenv import load_dotenv
import threading 
import queue 
import copy
import spacy

os.environ["GEMINI_API_KEY"] = "AIzaSyCveUj7czWR6CLHgzNGLiE6FZ4jMjFreYM"

def call_gemini(prompt, model="gemini-2.5-flash", temperature=0.7, max_output_tokens=512):
    api_key = os.getenv("GEMINI_API_KEY")
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key=AIzaSyCveUj7czWR6CLHgzNGLiE6FZ4jMjFreYM"

    headers = {"Content-Type": "application/json"}
    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {
            "temperature": temperature,
            "maxOutputTokens": max_output_tokens
        }
    }

    response = requests.post(url, headers=headers, json=payload)
    if response.status_code == 200:
        data = response.json()
        return data["candidates"][0]["content"]["parts"][0]["text"]
    else:
        raise Exception(f"Gemini API Error {response.status_code}: {response.text}")
    

import pandas as pd
import datetime
import re



class HeuristicEngine:
    def __init__(self):
        # Register rules: keyword → function
        self.rules = {
            "gender": self._infer_gender,
            "sex": self._infer_gender,
            "date": self._infer_date,
            "count": self._infer_count,
            "missing": self._infer_missing_values,
        }

    def apply(self, query: str, df: pd.DataFrame) -> str:
        """
        Main entry point: check the query text and apply the right heuristic.
        """
        query_lower = query.lower()

        for key, func in self.rules.items():
            if key in query_lower:
                return func(df, query)

        return "⚠️ No heuristic available for this type of query."

    # --------------------
    # Heuristic functions
    # --------------------

    def _infer_gender(self, df: pd.DataFrame, query: str) -> str:
        # 1) If dataset has gender/sex column → just confirm
        possible_cols = [c for c in df.columns if "gender" in c.lower() or "sex" in c.lower()]
        if possible_cols:
            return f"✅ Gender/sex column exists: {possible_cols[0]}"

        # 2) If not → try to guess from query
        match = re.search(r"name\s*=?\s*([a-zA-Z]+)", query.lower())
        if match:
            character_name = match.group(1).capitalize()
            guessed = self._guess_gender_from_name(character_name)
            return f"🤔 Guessed gender of {character_name}: {guessed}"

        return "⚠️ No gender/sex column found and could not guess."

    def _guess_gender_from_name(self, name: str) -> str:
        """
        Tiny heuristic to guess gender from first name.
        """
        if not name:
            return "Unknown"

        name_lower = name.lower()

        # Simple lookup dictionaries
        male_names = {"thor", "bruce", "steve", "tony", "peter", "clint", "scott", "vision"}
        female_names = {"natasha", "wanda", "gamora", "hope"}

        if name_lower in male_names:
            return "Male"
        if name_lower in female_names:
            return "Female"

        # Weak heuristic: names ending with 'a' → Female (with exceptions)
        if name_lower.endswith("a") and name_lower not in {"t'challa"}:
            return "Female"

        return "Unknown"

    def _infer_date(self, df: pd.DataFrame, query: str) -> str:
        # If dataset has a date column, point it out
        possible_cols = [c for c in df.columns if "date" in c.lower()]
        if possible_cols:
            return f"✅ Date column exists: {possible_cols[0]}"
        else:
            return f"📅 Today’s date is {datetime.date.today()}"

    def _infer_count(self, df: pd.DataFrame, query: str) -> str:
        return f"🧮 Dataset has {len(df)} rows and {len(df.columns)} columns."

    def _infer_missing_values(self, df: pd.DataFrame, query: str) -> str:
        missing = df.isnull().sum()
        missing_dict = {col: int(val) for col, val in missing.items() if val > 0}
        if not missing_dict:
            return "✅ No missing values detected."
        return f"⚠️ Missing values found: {missing_dict}"

# Load environment variables 

load_dotenv() 
@dataclass 
class ExecutionState: 
    """Tracks the state of code execution""" 
    variables: Dict[str, Any] = field(default_factory=dict) 
    imports: Set[str] = field(default_factory=set) 
    functions: Dict[str, str] = field(default_factory=dict) 
    data_artifacts: Dict[str, Any] = field(default_factory=dict) 
    exploration_results: Dict[str, Any] = field(default_factory=dict) 

@dataclass 

class GoalNode: 
    """Represents a goal in the goal decomposition tree""" 
    id: str 
    description: str 
    parent_id: Optional[str] = None 
    children: List[str] = field(default_factory=list) 
    status: str = "pending"  # pending, in_progress, completed, failed, blocked 
    priority: int = 1 
    dependencies: List[str] = field(default_factory=list) 
    result: Optional[str] = None 
    reasoning_chain: List[str] = field(default_factory=list) 
    execution_context: Dict[str, Any] = field(default_factory=dict) 
    retry_count: int = 0 
    max_retries: int = 5 
    complexity_score: float = 1.0

@dataclass

class AutonomousMemory:
    """Enhanced memory for smarter autonomous learning""" 
    execution_patterns: List[Dict[str, Any]] = field(default_factory=list) 
    successful_strategies: List[Dict[str, Any]] = field(default_factory=list) 
    installed_packages: Set[str] = field(default_factory=set) 
    context_discovered: Dict[str, Any] = field(default_factory=dict) 
    error_recovery_strategies: List[Dict[str, Any]] = field(default_factory=list) 
    goal_tree: Dict[str, GoalNode] = field(default_factory=dict) 
    reasoning_chains: List[List[str]] = field(default_factory=list) 
    exploration_cache: Dict[str, Any] = field(default_factory=dict) 
    execution_state: ExecutionState = field(default_factory=ExecutionState) 
    meta_learning: Dict[str, Any] = field(default_factory=dict) 

class ComplexityAnalyzer: 
    """Analyzes query complexity and suggests decomposition strategies""" 
    @staticmethod 
    def analyze_complexity(query: str) -> Dict[str, Any]: 
        """Analyze query complexity and return insights""" 
        complexity_indicators = { 
            'data_operations': ['analyze', 'process', 'clean', 'transform', 'merge', 'join', 'aggregate'], 
            'visualization': ['plot', 'chart', 'graph', 'visualize', 'show', 'display'], 
            'machine_learning': ['predict', 'classify', 'cluster', 'train', 'model', 'accuracy'], 
            'web_scraping': ['scrape', 'crawl', 'extract', 'website', 'url', 'web'], 
            'file_operations': ['file', 'csv', 'json', 'excel', 'read', 'write', 'save'], 
            'api_integration': ['api', 'request', 'endpoint', 'fetch', 'retrieve'], 
            'multi_step': ['and', 'then', 'after', 'next', 'finally', 'also'], 
            'comparative': ['compare', 'versus', 'vs', 'difference', 'contrast'], 
            'comprehensive': ['comprehensive', 'detailed', 'thorough', 'complete', 'full']
        }
        query_lower = query.lower() 
        detected_types = [] 
        complexity_score = 1.0 

        for category, keywords in complexity_indicators.items(): 
            if any(keyword in query_lower for keyword in keywords): 
                detected_types.append(category) 
                complexity_score += 0.5 

        # Additional complexity factors 
        if len(query.split()) > 20: 
            complexity_score += 0.3 
        if '?' in query and query.count('?') > 1: 
            complexity_score += 0.2 
        if any(word in query_lower for word in ['multiple', 'several', 'various', 'different']): 
            complexity_score += 0.4 

        return { 
            'complexity_score': min(complexity_score, 5.0), 
            'detected_types': detected_types, 
            'is_complex': complexity_score > 2.0, 
            'suggested_approach': ComplexityAnalyzer._suggest_approach(detected_types, complexity_score)
        }

    @staticmethod 
    def _suggest_approach(detected_types: List[str], complexity_score: float) -> str: 
        """Suggest the best approach based on complexity analysis""" 
        if complexity_score > 3.5: 
            return "hierarchical_decomposition"
        elif len(detected_types) > 3:
            return "parallel_execution"
        elif 'data_operations' in detected_types and len(detected_types) > 1: 
            return "pipeline_approach" 
        else:
            return "iterative_refinement"

class TrulyAutonomousAgent: 
    """ Enhanced TRULY autonomous agent with advanced capabilities for complex queries"""
    def __init__(self): 
        # Core components 
        self.execution_context = {} 
        self.session_history = [] 
        self.memory = AutonomousMemory() 
        self.max_iterations = 25 
        self.complexity_analyzer = ComplexityAnalyzer() 
        self.hueristics= HeuristicEngine()

        # Enhanced execution tracking 
        self.global_execution_state = ExecutionState() 
        self.exploration_depth = 3 
        self.adaptive_reasoning = True 
         
        # LLM configuration 
        self.azure_client = None 
        self.gemini_api_key = os.getenv('GEMINI_API_KEY') 
        self.ollama_host = None 
        self.model_name = "gemini-2.5-flash" 
        self.active_llm = "gemini" 

        # Enhanced prompting strategies 
        self.reasoning_strategies = { 
            'exploratory': self._create_exploratory_prompt, 
            'analytical': self._create_analytical_prompt, 
            'synthetic': self._create_synthetic_prompt,
            'adaptive': self._create_adaptive_prompt 
        }

        def __init__(self):
                self.nlp = spacy.load("en_core_web_sm")  # small English model

        def _nlp_preprocess(self, query: str) -> dict:
            """Process query with NLP to extract intent, entities, and numbers."""
            doc = self.nlp(query)

            entities = [(ent.text, ent.label_) for ent in doc.ents]
            numbers = [token.text for token in doc if token.like_num]
            keywords = [token.lemma_ for token in doc if token.is_alpha and not token.is_stop]

            return {
                "cleaned_text": doc.text,
                "entities": entities,
                "numbers": numbers,
                "keywords": keywords
            }

        # Setup LLM clients with fallback chain 
        self._setup_llm_clients()
        print(f"🤖 Enhanced Truly Autonomous Agent initialized") 
        print(f"🧠 Advanced Chain-of-Thought & Complex Query Handling enabled") 
        print(f"🔗 Active LLM: {self.active_llm}")

    def _setup_llm_clients(self):
        """Setup only Gemini client"""
        if not self.gemini_api_key:
            raise Exception("❌ GEMINI_API_KEY is missing. Please set it in your .env file.")

        try:
            # Test Gemini connection
            test_response = self._call_gemini("Test", max_tokens=5)
            if test_response:
                print(f"✅ Gemini connected successfully")
                self.active_llm = "Gemini"
            else:
                raise Exception("Gemini API did not return a response")
        except Exception as e:
            raise Exception(f"Gemini initialization failed: {e}")
        
    def _call_llm(self, prompt: str, temperature: float = 0.1, max_tokens: int = 2000) -> str:
        """Route all LLM calls to Gemini Flash"""
        try:
            return self._call_gemini(prompt, temperature, max_tokens)
        except Exception as e:
            print(f"❌ Gemini call failed: {e}")
            return None

    '''def _build_meta_prompt(self, user_query: str, csv_columns: list) -> str:
        """Constructs a meta-prompt that makes Gemini act as a dynamic heuristic engine generator."""
        return f"""
            You are an advanced data analysis assistant. When a user provides a file and asks a query, your task is to follow these steps strictly:

            1. Analyze the Request:
            - Determine if the needed information is present in the provided file schema: {csv_columns}.

            2. Handle Data Gaps with a Heuristic Engine:
            - If the information is NOT in the file, you MUST NOT pretend it is.
            - Instead, dynamically create a temporary, single-use "heuristic engine" based on your general knowledge or training dataset to find the answer.
            - Announce its creation with: Engine Name, Input, Logic/Rules, Output.

            3. Execute and Answer:
            - State that you are executing the engine with the specific query parameters.
            - Provide the result.
            - Finally, give the answer in natural language that should be clearnd in a conversational manner.

            ---
            User Query: "{user_query}"
            ---
            """
    '''
    def _call_gemini(self, prompt: str, temperature: float = 0.1, max_tokens: int = 2000) -> str:
        """Call Gemini API with enhanced error handling"""

        url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash-latest:generateContent?key={self.gemini_api_key}"

        headers = {
            "Content-Type": "application/json"
        }

        payload = {
            "contents": [
                {
                    "parts": [
                        {
                            "text": prompt
                        }
                    ]
                }
            ],
            "generationConfig": {
                "temperature": temperature,
                "maxOutputTokens": max_tokens
            }
        }

        response = requests.post(url, headers=headers, json=payload)



        if response.status_code == 200:
            data = response.json()
            return data["candidates"][0]["content"]["parts"][0]["text"]
        else:
            raise Exception(f"Gemini API Error {response.status_code}: {response.text}")


     

    def _call_ollama(self, prompt: str, temperature: float = 0.1, max_tokens: int = 2000) -> str: 
        """Call Ollama API with enhanced parameters""" 
        payload = { 
            "model": self.model_name, 
            "prompt": prompt, 
            "stream": False, 
            "options": { 
                "temperature": temperature, 
                "top_p": 0.9, 
                "num_predict": max_tokens, 
                "repeat_penalty": 1.1 
            }
        }

        response = requests.post( 
            f"{self.ollama_host}/api/generate", 
            json=payload, 
            timeout=120 
        )

        if response.status_code == 200: 
            return response.json()['response'].strip() 
        return None

    def _install_package_safely(self, package_name: str) -> bool: 
        """Enhanced package installation with better error handling""" 
        if package_name in self.memory.installed_packages: 
            return True

        try: 
            print(f"🔧 Installing package: {package_name}")
            # Handle special package mappings 

            package_mappings = {
                'sklearn': 'scikit-learn', 
                'cv2': 'opencv-python', 
                'PIL': 'Pillow',
                'bs4': 'beautifulsoup4' 
            }

            actual_package = package_mappings.get(package_name, package_name) 
            result = subprocess.run(
                [sys.executable, '-m', 'pip', 'install', actual_package], 
                capture_output=True, 
                text=True,
                timeout=180 
            ) 

            if result.returncode == 0: 
                self.memory.installed_packages.add(package_name) 
                self.memory.installed_packages.add(actual_package) 
                print(f"✅ Successfully installed {actual_package}") 
                return True

            else:
                print(f"❌ Failed to install {actual_package}: {result.stderr}") 
                return False

        except Exception as e: 
            print(f"❌ Installation error for {package_name}: {e}") 
            return False


    def _adaptive_goal_decomposition(self, user_goal: str) -> Dict[str, GoalNode]: 
        """Advanced goal decomposition with complexity analysis"""
        complexity_analysis = self.complexity_analyzer.analyze_complexity(user_goal) 
        context = self._build_comprehensive_context() 
        print(f"🔍 Complexity Analysis: {complexity_analysis['complexity_score']:.1f}/5.0") 
        print(f"📊 Detected Types: {', '.join(complexity_analysis['detected_types'])}") 
        print(f"🎯 Suggested Approach: {complexity_analysis['suggested_approach']}") 
        if complexity_analysis['is_complex']:
            return self._hierarchical_decomposition(user_goal, complexity_analysis, context) 
        else: 
            return self._simple_goal_creation(user_goal, complexity_analysis) 

    def _hierarchical_decomposition(self, user_goal: str, complexity_analysis: Dict, context: str) -> Dict[str, GoalNode]: 
        """Create hierarchical goal decomposition for complex queries""" 
        # Extract parameters from the original goal 
        goal_params = self._extract_goal_parameters(user_goal) 
        params_str = "" 
        if goal_params: 
            params_str = f"\nEXTRACTED PARAMETERS FROM GOAL: {goal_params}" 
        decomposition_prompt = f""" 

You are an advanced autonomous problem-solving agent. Decompose this complex goal into manageable sub-goals. 
GOAL: {user_goal}
COMPLEXITY SCORE: {complexity_analysis['complexity_score']:.1f}/5.0 
DETECTED TYPES: {complexity_analysis['detected_types']} 
SUGGESTED APPROACH: {complexity_analysis['suggested_approach']} 
{params_str} 
CONTEXT: 
{context}
DECOMPOSITION RULES: 
1. Break down into 2-5 logical sub-goals 
2. Each sub-goal should be independently executable 
3. Consider data dependencies between goals
4. Include exploration/discovery phases for data-related tasks 
5. Plan for iterative refinement and validation 
6. PRESERVE all user-specified parameters (file paths, URLs, etc.) in goal descriptions 
7. Each goal description should reference the EXACT parameters from the original goal 
IMPORTANT: When creating goal descriptions, include the EXACT file paths, URLs, and parameters from the original goal. Do not use generic placeholders like "data.csv" or "file.txt". 

Respond in this EXACT format: 

REASONING: [Your analysis of why this decomposition is needed and how to preserve parameters] 
GOALS: 
1. GOAL_ID: exploration | DESCRIPTION: [Description using exact parameters] | PRIORITY: [1-5] | DEPENDENCIES: [none or goal_ids] 
2. GOAL_ID: preparation | DESCRIPTION: [Description using exact parameters] | PRIORITY: [1-5] | DEPENDENCIES: [goal_ids] 
3. GOAL_ID: execution | DESCRIPTION: [Description using exact parameters] | PRIORITY: [1-5] | DEPENDENCIES: [goal_ids] 
4. GOAL_ID: validation | DESCRIPTION: [Description using exact parameters] | PRIORITY: [1-5] | DEPENDENCIES: [goal_ids] 

REASONING:""" 
        response = self._call_llm(decomposition_prompt, temperature=0.2, max_tokens=2000)
        if not response: 
            return self._simple_goal_creation(user_goal, complexity_analysis) 
        return self._parse_goal_decomposition(response, complexity_analysis['complexity_score']) 


    def _parse_goal_decomposition(self, response: str, complexity_score: float) -> Dict[str, GoalNode]: 
        """Parse the goal decomposition response into GoalNode objects""" 
        goals = {} 
        # Extract goals from response 
        goals_section = re.search(r'GOALS:\s*\n(.*?)(?:\n\n|$)', response, re.DOTALL) 
        if not goals_section: 
            # Fallback to simple goal 
            return {"goal_1": GoalNode(id="goal_1", description="Execute comprehensive task", priority=1, complexity_score=complexity_score)} 
        goal_lines = goals_section.group(1).strip().split('\n') 
        for line in goal_lines: 
            if not line.trip() or not line.strip()[0].isdigit(): 
                continue 
            # Parse goal line
            goal_match = re.search(r'GOAL_ID:\s*(\w+)\s*\|\s*DESCRIPTION:\s*([^|]+)\s*\|\s*PRIORITY:\s*(\d+)\s*\|\s*DEPENDENCIES:\s*([^|]*)', line) 
            if goal_match: 
                goal_id = goal_match.group(1).strip() 
                description = goal_match.group(2).strip() 
                priority = int(goal_match.group(3).strip()) 
                deps_text = goal_match.group(4).strip() 
                 
                dependencies = [] 
                if deps_text and deps_text.lower() != 'none': 
                    dependencies = [dep.strip() for dep in deps_text.split(',') if dep.strip()] 
                 
                goals[goal_id] = GoalNode( 
                    id=goal_id, 
                    description=description, 
                    priority=priority, 
                    dependencies=dependencies, 
                    complexity_score=complexity_score / len(goal_lines) 
                ) 
        

        if not goals: 
            # Fallback 
            return {"goal_1": GoalNode(id="goal_1", description="Execute comprehensive task", priority=1, complexity_score=complexity_score)} 
         
        return goals 
     
    def _simple_goal_creation(self, user_goal: str, complexity_analysis: Dict) -> Dict[str, GoalNode]: 
        """Create a single comprehensive goal for simpler queries""" 
        goal_id = "goal_1" 
        return {goal_id: GoalNode( 
            id=goal_id, 
            description=user_goal, 
            priority=1, 
            complexity_score=complexity_analysis['complexity_score'] 
        )} 
     
    def _get_next_goal(self) -> Optional[GoalNode]: 
        """Enhanced goal selection with dependency resolution""" 
        available_goals = [] 
         
        for goal in self.memory.goal_tree.values(): 
            if goal.status == "pending": 
                # Check if all dependencies are completed 
                dependencies_met = True 
                for dep_id in goal.dependencies: 
                    if dep_id in self.memory.goal_tree: 
                        if self.memory.goal_tree[dep_id].status != "completed": 
                            dependencies_met = False 
                            break 
                    else: 
                        dependencies_met = False 
                        break 
                 
                if dependencies_met and goal.retry_count < goal.max_retries: 
                    available_goals.append(goal) 
         
        if not available_goals: 
            return None 
         
        # Sort by priority, then by complexity 
        return min(available_goals, key=lambda g: (g.priority, -g.complexity_score))
     
    def _extract_goal_parameters(self, goal_description: str) -> Dict[str, Any]: 
        """Extract specific parameters from goal description (file paths, URLs, etc.)""" 
        parameters = {} 
         
        # Extract file paths (both relative and absolute) 
        file_path_patterns = [ 
            r'([/\\][\w\s\-_.\\\/]+\.(?:csv|txt|json|xlsx|xls|pdf|doc|docx))',  # Absolute paths 
            r'([\w\s\-_.]+\.(?:csv|txt|json|xlsx|xls|pdf|doc|docx))',  # Relative paths with extensions 
            r'analyze\s+([^\s]+\.(?:csv|txt|json|xlsx|xls))',  # "analyze filename.csv" 
            r'file\s+([^\s]+\.(?:csv|txt|json|xlsx|xls))',     # "file filename.csv" 
            r'read\s+([^\s]+\.(?:csv|txt|json|xlsx|xls))',     # "read filename.csv" 
        ] 
         
        for pattern in file_path_patterns: 
            matches = re.findall(pattern, goal_description, re.IGNORECASE) 
            if matches: 
                # Take the longest/most complete path found 
                file_path = max(matches, key=len) if isinstance(matches[0], str) else matches[0] 
                parameters['file_path'] = file_path.strip() 
                break 
         
        # Extract URLs 
        url_pattern = r'(https?://[^\s]+)' 
        urls = re.findall(url_pattern, goal_description) 
        if urls:
            parameters['urls'] = urls 
         
        # Extract numbers/counts for context 
        number_pattern = r'\b(\d+)\b' 
        numbers = re.findall(number_pattern, goal_description)
        if numbers: 
            parameters['numbers'] = [int(n) for n in numbers] 
         
        # Extract column names mentioned 
        column_pattern = r'column[s]?\s+["\']?([^"\']+)["\']?' 
        columns = re.findall(column_pattern, goal_description, re.IGNORECASE) 
        if columns:
            parameters['columns'] = columns 

        return parameters 

 
    def _create_exploratory_prompt(self, goal: GoalNode, iteration: int, context: str) -> str: 
        """Create exploratory reasoning prompt for discovery phases""" 
         
        # Extract parameters from the goal description 
        goal_params = self._extract_goal_parameters(goal.description) 
        params_str = "" 
        if goal_params: 
            params_str = f"\nEXTRACTED PARAMETERS: {goal_params}"

        return f""" 
You are an advanced autonomous exploration agent. Your task is to EXPLORE and DISCOVER before taking action. 
 
GOAL: {goal.description} 
ITERATION: {iteration}/3 
{params_str} 
 
EXECUTION CONTEXT: 
{context}

INSTALLED PACKAGES: {list(self.memory.installed_packages)} 
EXECUTION STATE: {self._summarize_execution_state()} 
 
CRITICAL AUTONOMOUS REASONING RULES: 
- NEVER hardcode file paths, URLs, or parameters 
- ALWAYS use the EXACT file paths, URLs, and parameters specified in the goal 
- ANALYZE the data context to understand what needs to be counted/analyzed 
- If goal mentions "count characters" in context of a dataset, determine from column headers what "characters" means 
- If you see column names like "character_name", "person_name", "movie_title", etc., understand these are RECORDS to count 
- Be context-aware: counting in a dataset usually means counting rows/records, not text characters 
 
EXPLORATION STRATEGY: 
1. If dealing with data: First examine structure, columns, data types, missing values, missing columns
2. If dealing with files: Check existence, format, size, accessibility - USE EXACT FILE PATHS 
3. If dealing with APIs: Test endpoints, check authentication, verify responses 
4. Use defensive programming - always check before assuming 
5. Extract and preserve all user-specified parameters (file paths, URLs, etc.) 
6. ANALYZE COLUMN HEADERS to understand what the user wants to count
 
Generate Python code that EXPLORES the problem space and UNDERSTANDS the context: 
EXPLORATION_REASONING: [Why exploration is needed, what to discover, and how to interpret "count" in this context] 
DISCOVERY_PLAN: [Step-by-step exploration approach using extracted parameters] 
PACKAGES_NEEDED: [List packages to install] 
EXPLORATION_CODE: [Python code that explores, discovers, and analyzes context using EXACT parameters from goal] 
 
EXPLORATION_REASONING:""" 
     
    def _create_analytical_prompt(self, goal: GoalNode, iteration: int, context: str) -> str: 
        """Create analytical reasoning prompt for problem-solving phases""" 
         
        # Extract parameters from the goal description 
        goal_params = self._extract_goal_parameters(goal.description) 
        params_str = "" 
        if goal_params: 
            params_str = f"\nEXTRACTED PARAMETERS: {goal_params}" 
         
        return f""" 
You are an advanced autonomous analytical agent. Analyze the problem deeply and solve systematically. 
 
GOAL: {goal.description} 
ITERATION: {iteration}/3 
{params_str} 

EXECUTION CONTEXT:
{context} 

EXPLORATION RESULTS: {self._get_exploration_summary()}
EXECUTION STATE: {self._summarize_execution_state()} 

 
CRITICAL AUTONOMOUS REASONING RULES:
- NEVER hardcode file paths, URLs, or any parameters 
- ALWAYS use the EXACT file paths, URLs, and parameters specified in the original goal 
- Reference extracted parameters directly in your code 
- Preserve user specifications exactly as provided 
- ANALYZE THE CONTEXT: If counting in a structured dataset, usually means counting records/rows 
- Look at column headers and data structure to understand what should be counted 
- If you see data with rows and columns, "count characters" likely means count records (rows) 
 
ANALYTICAL APPROACH: 
1. Leverage discovered information from exploration 
2. Break down the problem into logical steps 
3. Consider edge cases and potential errors 
4. Use robust error handling 
5. Validate results and provide insights 
6. Use EXACT parameters from the original goal (file paths, URLs, etc.) 
7. Apply CONTEXTUAL REASONING to interpret what needs to be counted 
 
Generate comprehensive Python code: 
 
ANALYTICAL_REASONING: [Deep analysis of the problem, approach, and what "counting" means in this context] 
SOLUTION_STRATEGY: [Systematic solving approach using exact parameters and context understanding] 
PACKAGES_NEEDED: [List packages to install] 
ANALYTICAL_CODE: [Complete Python solution with error handling using EXACT parameters and context-aware counting] 



ANALYTICAL_REASONING:"""
     
    def _create_synthetic_prompt(self, goal: GoalNode, iteration: int, context: str) -> str:
        """Create synthetic reasoning prompt for integration phases""" 
         
        # Extract parameters from the goal description 
        goal_params = self._extract_goal_parameters(goal.description) 
        params_str = "" 
        if goal_params:
            params_str = f"\nEXTRACTED PARAMETERS: {goal_params}"
        return f""" 
You are an advanced autonomous synthesis agent. Integrate multiple components into a cohesive solution. 
 
GOAL: {goal.description} 
ITERATION: {iteration}/3 
{params_str} 
 
EXECUTION CONTEXT: 
{context} 
 
AVAILABLE COMPONENTS: {self._get_available_components()} 
EXECUTION STATE: {self._summarize_execution_state()}
 
PARAMETER PRESERVATION REQUIREMENTS:
- Use EXACT file paths, URLs, and parameters from the original goal 
- Never substitute or modify user-specified paths/parameters 
- Reference extracted parameters directly 

SYNTHESIS APPROACH: 
1. Combine available data, functions, and insights 
2. Create integrated workflows using exact user parameters 
3. Ensure consistency across components 
4. Optimize performance and output quality 
5. Provide comprehensive results 

Generate integrated Python code:
SYNTHESIS_REASONING: [How to integrate available components] 
INTEGRATION_STRATEGY: [Approach to combine and optimize using exact parameters] 
PACKAGES_NEEDED: [List packages to install]
SYNTHESIS_CODE: [Integrated Python solution using EXACT user parameters]
SYNTHESIS_REASONING:"""
     
    def _create_adaptive_prompt(self, goal: GoalNode, iteration: int, context: str) -> str: 
        """Create adaptive reasoning prompt that learns from previous attempts""" 
         
        # Extract parameters from the goal description 
        goal_params = self._extract_goal_parameters(goal.description) 
        params_str = "" 
        if goal_params: 
            params_str = f"\nEXTRACTED PARAMETERS: {goal_params}" 
         
        return f""" 
You are an advanced autonomous adaptive agent. Learn from previous attempts and adapt your approach. 
 
GOAL: {goal.description} 
ITERATION: {iteration}/3 
RETRY COUNT: {goal.retry_count} 
{params_str} 
 
EXECUTION CONTEXT: 
{context} 
 
LEARNING FROM FAILURES: {self._get_failure_analysis()} 
EXECUTION STATE: {self._summarize_execution_state()} 
 
CRITICAL ADAPTATION RULES: 
- NEVER hardcode file paths or parameters - use EXACT ones from the goal 
- Learn from previous errors while preserving user-specified parameters 
- If file path errors occurred, verify you're using the EXACT path from the goal 
- Adapt methods, not parameters 
 
ADAPTIVE STRATEGY: 
1. Analyze what went wrong in previous attempts 
2. Identify patterns in errors and failures 
3. Adapt approach based on learned lessons while preserving exact parameters 
4. Use alternative methods when standard approaches fail 
5. Implement robust fallback mechanisms 
6. Always use EXACT file paths, URLs, and parameters from the original goal

Generate improved Python code:
ADAPTIVE_REASONING: [Learning from previous attempts and adaptation strategy] 
IMPROVED_APPROACH: [How this attempt differs from previous failures while preserving parameters] 
PACKAGES_NEEDED: [List packages to install] 
ADAPTIVE_CODE: [Improved Python solution with lessons learned using EXACT parameters]
ADAPTIVE_REASONING:""" 

     

    def _enhanced_autonomous_reasoning(self, goal: GoalNode, iteration: int) -> Dict[str, Any]:
        """Enhanced reasoning that selects appropriate strategy based on context""" 

        context = self._build_comprehensive_context() 

        # Select reasoning strategy based on goal and iteration 
        if 'exploration' in goal.id.lower() or iteration == 1: 
            strategy = 'exploratory' 
        elif goal.retry_count > 2: 
            strategy = 'adaptive' 
        elif len(self.global_execution_state.data_artifacts) > 0: 
            strategy = 'synthetic' 
        else: 
            strategy = 'analytical' 

        print(f"🧠 Using {strategy} reasoning strategy")

        '''# 🔍 Build schema-aware meta-prompt
        if hasattr(self, "active_dataframe") and self.active_dataframe is not None:
            csv_columns = self.active_dataframe.columns.tolist()
        else:
            csv_columns = []

        prompt = self._build_meta_prompt(goal.description, csv_columns)
        '''
        prompt = self.reasoning_strategies[strategy](goal, iteration, context)

        response = self._call_llm(prompt, temperature=0.15, max_tokens=3000) 


        if not response: 
            return {
                "reasoning": "No response from LLM",
                "strategy": "retry",
                "packages_needed": [],
                "code": ""
            }

        return self._parse_enhanced_response(response, strategy) 

     

    def _parse_enhanced_response(self, response: str, strategy: str) -> Dict[str, Any]: 

        """Parse enhanced response based on strategy""" 

         

        # Strategy-specific parsing 

        if strategy == 'exploratory': 

            reasoning_key = 'EXPLORATION_REASONING' 

            strategy_key = 'DISCOVERY_PLAN' 

            code_key = 'EXPLORATION_CODE' 

        elif strategy == 'analytical': 

            reasoning_key = 'ANALYTICAL_REASONING' 

            strategy_key = 'SOLUTION_STRATEGY' 

            code_key = 'ANALYTICAL_CODE' 

        elif strategy == 'synthetic': 

            reasoning_key = 'SYNTHESIS_REASONING' 

            strategy_key = 'INTEGRATION_STRATEGY' 

            code_key = 'SYNTHESIS_CODE' 

        else:  # adaptive 

            reasoning_key = 'ADAPTIVE_REASONING' 

            strategy_key = 'IMPROVED_APPROACH' 

            code_key = 'ADAPTIVE_CODE' 

         

        # Parse response sections 

        reasoning_match = re.search(f'{reasoning_key}:\\s*(.*?)(?={strategy_key}:|PACKAGES_NEEDED:|$)', response, re.DOTALL) 

        reasoning = reasoning_match.group(1).strip() if reasoning_match else "No reasoning provided" 

         

        strategy_match = re.search(f'{strategy_key}:\\s*(.*?)(?=PACKAGES_NEEDED:|{code_key}:|$)', response, re.DOTALL) 

        strategy_text = strategy_match.group(1).strip() if strategy_match else "execute" 

         

        packages_match = re.search(r'PACKAGES_NEEDED:\\s*(.*?)(?=' + code_key + ':|$)', response, re.DOTALL) 

        packages_text = packages_match.group(1).strip() if packages_match else "" 

         

        # Extract package names with better pattern matching 

        packages_needed = [] 

        if packages_text and packages_text.lower() != "none": 

            # Try multiple patterns 

            package_patterns = re.findall(r'[\'\"]([\w-]+)[\'\"]', packages_text) 

            if not package_patterns: 

                package_patterns = re.findall(r'\b(pandas|numpy|matplotlib|seaborn|requests|beautifulsoup4|scipy|sklearn|plotly|streamlit|fastapi|flask|django)\b', packages_text.lower()) 

            if not package_patterns: 

                # Simple word extraction 

                words = packages_text.replace(',', ' ').replace('\n', ' ').split() 

                package_patterns = [w for w in words if w.isalnum() and len(w) > 2] 

            packages_needed = [p for p in package_patterns if p not in ['None', 'none', 'install', 'pip']] 

         

        code = self._extract_enhanced_code(response, code_key) 

         

        return { 

            "reasoning": reasoning, 

            "strategy": strategy_text, 

            "packages_needed": packages_needed, 

            "code": code 

        } 

     

    def _extract_enhanced_code(self, response: str, code_key: str) -> str: 

        """Enhanced code extraction with multiple patterns""" 

        if not response: 

            return "" 

         

        patterns = [ 

            f'{code_key}:\\s*\\n```python\\s*\\n(.*?)\\n```', 

            f'{code_key}:\\s*\\n```\\s*\\n(.*?)\\n```', 

            f'{code_key}:\\s*\\n(.*?)(?=\\n\\n|$)', 

            r'```python\s*\n(.*?)\n```', 

            r'```\s*\n(.*?)\n```' 

        ] 

         

        for pattern in patterns: 

            match = re.search(pattern, response, re.DOTALL) 

            if match: 

                code = match.group(1).strip() 

                if len(code) > 10:  # Ensure we got substantial code 

                    return code 

         

        return "" 

     

    def _build_comprehensive_context(self) -> str: 

        """Build comprehensive execution context""" 

        context_parts = [] 

         

        # Recent execution summary 

        if self.session_history: 

            recent_attempts = len(self.session_history[-5:]) 

            recent_successes = len([r for r in self.session_history[-5:] if r.get('success')]) 

            context_parts.append(f"RECENT PERFORMANCE: {recent_successes}/{recent_attempts} successes") 

         

        # Available data and artifacts 

        if self.global_execution_state.data_artifacts: 

            context_parts.append("AVAILABLE DATA ARTIFACTS:") 

            for name, info in list(self.global_execution_state.data_artifacts.items())[:3]: 

                context_parts.append(f"  • {name}: {str(info)[:100]}") 

         

        # Available functions and variables 

        if self.global_execution_state.variables: 

            var_count = len(self.global_execution_state.variables) 

            context_parts.append(f"AVAILABLE VARIABLES: {var_count} variables in scope") 

         

        # Recent errors with learning 

        recent_errors = [] 

        for record in self.session_history[-3:]: 

            if not record.get('success') and record.get('error'): 

                error_summary = record['error'].split('\n')[0][:150] 

                recent_errors.append(error_summary) 

         

        if recent_errors: 

            context_parts.append("RECENT ERRORS TO LEARN FROM:") 

            for i, error in enumerate(recent_errors, 1): 

                context_parts.append(f"  {i}. {error}") 

         

        return "\n".join(context_parts) if context_parts else "Starting fresh with no prior context." 

     

    def _summarize_execution_state(self) -> str: 

        """Summarize current execution state""" 

        state_summary = [] 

         

        if self.global_execution_state.variables: 

            state_summary.append(f"Variables: {len(self.global_execution_state.variables)}") 

        if self.global_execution_state.data_artifacts: 

            state_summary.append(f"Data artifacts: {len(self.global_execution_state.data_artifacts)}") 

        if self.global_execution_state.functions: 

            state_summary.append(f"Functions: {len(self.global_execution_state.functions)}") 

         

        return ", ".join(state_summary) if state_summary else "Empty state" 

     

    def _get_exploration_summary(self) -> str: 

        """Get summary of exploration results""" 

        if not self.global_execution_state.exploration_results: 

            return "No exploration results available" 

         

        summary = [] 

        for key, value in list(self.global_execution_state.exploration_results.items())[:3]: 

            summary.append(f"{key}: {str(value)[:100]}") 

         

        return "; ".join(summary) 

     

    def _get_available_components(self) -> str: 

        """Get summary of available components for synthesis""" 

        components = [] 

         

        if self.global_execution_state.data_artifacts: 

            components.append(f"Data: {list(self.global_execution_state.data_artifacts.keys())}") 

        if self.global_execution_state.functions: 

            components.append(f"Functions: {list(self.global_execution_state.functions.keys())}") 

        if self.global_execution_state.variables: 

            var_names = [k for k in self.global_execution_state.variables.keys() if not k.startswith('_')][:5] 

            components.append(f"Variables: {var_names}") 

         

        return "; ".join(components) if components else "No components available" 

     

    def _get_failure_analysis(self) -> str: 

        """Analyze previous failures for adaptive learning""" 

        failures = [r for r in self.session_history[-5:] if not r.get('success')] 

         

        if not failures: 

            return "No recent failures to learn from" 

         

        error_patterns = [] 

        for failure in failures[-3:]: 

            if failure.get('error'): 

                error_type = failure['error'].split(':')[0] if ':' in failure['error'] else 'Unknown' 

                error_patterns.append(error_type) 

         

        pattern_counts = {} 

        for pattern in error_patterns: 

            pattern_counts[pattern] = pattern_counts.get(pattern, 0) + 1 

         

        analysis = f"Common errors: {pattern_counts}" 

        return analysis 

     

    def _execute_code_with_state_tracking(self, code: str, goal: GoalNode) -> Tuple[str, str, bool]:
        """Execute code with enhanced state tracking"""

        if not code:
            return "", "No code provided", False

        stdout_capture = io.StringIO()
        stderr_capture = io.StringIO()

        try:
            exec_globals = {
                '__builtins__': __builtins__,
                'subprocess': subprocess,
                'sys': sys,
                'os': os,
                'importlib': importlib,
                'print': print,  # Ensure print works
            }

            # Add existing context and state
            exec_globals.update(self.execution_context)
            exec_globals.update(self.global_execution_state.variables)

            # Execute code with state capture
            with contextlib.redirect_stdout(stdout_capture), contextlib.redirect_stderr(stderr_capture):
                compiled_code = compile(code, '<string>', 'exec')
                exec(compiled_code, exec_globals)

            # Update global execution state
            self._update_execution_state(exec_globals, goal)

            # Collect results
            stdout_text = stdout_capture.getvalue().strip()
            stderr_text = stderr_capture.getvalue().strip()
            success = (stderr_text == "")

            # 🔄 Heuristics fallback (only if output suggests missing data)
            if ("Error" in stdout_text or "missing" in stdout_text.lower() or "not found" in stdout_text.lower()):
                try:
                    heuristics_result = self.hueristics.apply(goal.description, exec_globals.get("df", None))
                    stdout_text += f"\n[Heuristics Applied] {heuristics_result}"
                except Exception as he:
                    stdout_text += f"\n[Heuristics Failed] {str(he)}"

            # ✅ Save into session history for evaluation
            self.session_history.append({
                "goal_id": goal.id,
                "success": success,
                "output": stdout_text if stdout_text else stderr_text
            })

            return stdout_text, stderr_text, success

        except Exception as e:
            error_msg = f"{type(e).__name__}: {str(e)}\n{traceback.format_exc()}"
            stdout_text = stdout_capture.getvalue().strip()
            stderr_text = error_msg

            # Save failure into session history
            self.session_history.append({
                "goal_id": goal.id,
                "success": False,
                "output": error_msg
            })

            print(f"⚠️ Code execution failed: {e}")
            return stdout_text, stderr_text, False


        

    def _update_execution_state(self, exec_globals: Dict, goal: GoalNode): 

        """Update global execution state with new variables and artifacts""" 

         

        # Update variables (excluding built-ins and imports) 

        for key, value in exec_globals.items(): 

            if not key.startswith('__') and key not in ['subprocess', 'sys', 'os', 'importlib', 'print']: 

                self.global_execution_state.variables[key] = value 

                self.execution_context[key] = value 

                 

                # Detect data artifacts 

                if hasattr(value, 'shape') and hasattr(value, 'columns'):  # DataFrame 

                    self.global_execution_state.data_artifacts[key] = { 

                        'type': 'DataFrame', 

                        'shape': getattr(value, 'shape', None), 

                        'columns': list(getattr(value, 'columns', [])), 

                        'dtypes': str(getattr(value, 'dtypes', None)) 

                    } 

                elif hasattr(value, 'shape') and hasattr(value, 'dtype'):  # NumPy array 

                    self.global_execution_state.data_artifacts[key] = { 

                        'type': 'Array', 

                        'shape': getattr(value, 'shape', None), 

                        'dtype': str(getattr(value, 'dtype', None)) 

                    } 

                elif isinstance(value, (list, tuple)) and len(value) > 0: 

                    self.global_execution_state.data_artifacts[key] = { 

                        'type': type(value).__name__, 

                        'length': len(value), 

                        'sample': str(value[:3]) if len(value) > 3 else str(value) 

                    } 

                elif isinstance(value, dict) and len(value) > 0: 

                    self.global_execution_state.data_artifacts[key] = { 

                        'type': 'dict', 

                        'keys': list(value.keys())[:5], 

                        'length': len(value) 

                    } 

         

        # Store exploration results for specific goal types 

        if 'exploration' in goal.id.lower(): 

            goal_results = {} 

            for key, value in exec_globals.items(): 

                if not key.startswith('__') and key not in ['subprocess', 'sys', 'os', 'importlib']: 

                    if isinstance(value, (str, int, float, bool, list, dict)): 

                        goal_results[key] = value 

             

            self.global_execution_state.exploration_results.update(goal_results) 

     

    def _enhanced_goal_evaluation(self, goal: GoalNode) -> Dict[str, Any]:
        """Enhanced goal completion evaluation with better answer extraction""" 
        successful_outputs = [] 
        for record in self.session_history[-3:]: 
            if record.get('success') and record.get('output') and record.get('goal_id') == goal.id: 
                successful_outputs.append(record['output']) 


        if not successful_outputs: 
            return { 
                "completed": False, 
                "confidence": 0, 
                "reason": "No successful outputs yet", 
                "extracted_answer": None 
            } 
         

        # Enhanced evaluation logic with answer extraction 
        output_text = " ".join(successful_outputs).lower() 
        latest_output = successful_outputs[-1]

        # Extract key answers based on goal type and content 
        extracted_answer = self._extract_key_answer(goal, latest_output, output_text) 

        # Goal-specific evaluation 
        if 'exploration' in goal.id.lower(): 
            exploration_indicators = ['shape', 'columns', 'dtype', 'info', 'head', 'describe', 'found', 'discovered', 'headers', 'encoding'] 
            if any(indicator in output_text for indicator in exploration_indicators): 
                return { 
                    "completed": True, 
                    "confidence": 90,
                    "reason": "Exploration phase completed successfully", 
                    "extracted_answer": extracted_answer
                }

        elif 'preparation' in goal.id.lower() or 'prepare' in goal.id.lower(): 
            prep_indicators = ['cleaned', 'processed', 'transformed', 'prepared', 'ready', 'dataframe', 'loaded'] 
            if any(indicator in output_text for indicator in prep_indicators):
                return {
                    "completed": True, 
                    "confidence": 85, 
                    "reason": "Preparation phase completed", 
                    "extracted_answer": extracted_answer 
                }
         
        elif 'count' in goal.id.lower() or 'count' in goal.description.lower(): 
            # Look for counting results 
            count_indicators = ['total', 'count', 'rows', 'shape', 'number of'] 
            if any(indicator in output_text for indicator in count_indicators): 
                return { 
                    "completed": True, 
                    "confidence": 95, 
                    "reason": "Counting completed with results", 
                    "extracted_answer": extracted_answer 
                }

        elif 'validat' in goal.id.lower(): 
            validation_indicators = ['valid', 'confirmed', 'accurate', 'verified', 'missing values', 'data types'] 
            if any(indicator in output_text for indicator in validation_indicators): 
                return { 
                    "completed": True, 
                    "confidence": 85, 
                    "reason": "Validation phase completed", 
                    "extracted_answer": extracted_answer 
                }

        # General success indicators 
        success_indicators = ['total', 'count', 'rows', 'characters', 'complete', 'finished', 'done', 'result', 'success', 'shape'] 
        if any(indicator in output_text for indicator in success_indicators): 
            return { 
                "completed": True, 
                "confidence": 80, 
                "reason": "Output contains expected results", 
                "extracted_answer": extracted_answer
            } 


        # Check for data artifacts creation 
        if len(self.global_execution_state.data_artifacts) > 0: 
            return { 
                "completed": True, 
                "confidence": 75, 
                "reason": "Data artifacts created successfully", 
                "extracted_answer": extracted_answer 
            }
        
        # Detect dynamic heuristic engine output
        if "Heuristic Engine Definition" in output_text or "Running Engine" in output_text:
            return {
                "completed": True,
                "confidence": 95,
                "reason": "Query answered using a dynamic heuristic engine",
                "extracted_answer": extracted_answer or latest_output
            }

         
        # Confidence threshold for final answer clarity
        confidence_score = max(0, len(successful_outputs) * 20)
        if confidence_score >= 70:
            return {
                "completed": True,
                "confidence": confidence_score,
                "reason": "Final answer confirmed. No clarification needed.",
                "extracted_answer": extracted_answer
            }
        else:
            return {
                "completed": False,
                "confidence": confidence_score,
                "reason": "Analysis completed but final answer needs clarification.",
                "extracted_answer": extracted_answer
            }

        '''return {
            "completed": False, 
            "confidence": max(0, len(successful_outputs)*20), 
            "reason": "Could not evaluate completion", 
            "extracted_answer": extracted_answer 
        }'''

    def _extract_key_answer(self, goal: GoalNode, output: str, output_lower: str) -> Optional[str]: 
        """Extract the key answer from goal output based on goal type and context""" 
         
        # For counting goals - extract the count with autonomous context understanding 
        if 'count' in goal.description.lower() or 'count' in goal.id.lower(): 
             
            # Autonomous reasoning: If we see structured data patterns, counting likely means records 
            has_structured_data_indicators = any(indicator in output_lower for indicator in [ 
                'column', 'header', 'row', 'dataframe', 'csv', 'records', 'entries' 
            ]) 
             
            if has_structured_data_indicators: 
                # Look for line count patterns (common in CSV analysis) 
                line_patterns = [ 
                    r'number of lines:\s*(\d+)',
                    r'line count:\s*(\d+)',  
                    r'(\d+)\s+lines', 
                    r'total lines:\s*(\d+)' 
                ] 
                 
                for pattern in line_patterns: 
                    match = re.search(pattern, output_lower) 
                    if match: 
                        total_lines = int(match.group(1)) 
                        # Autonomous reasoning: If there are headers mentioned, subtract 1 
                        has_header = any(h in output_lower for h in ['header', 'column names', 'columns:']) 
                        if has_header and total_lines > 1: 
                            record_count = total_lines - 1 
                            return f"Records found: {record_count} (from {total_lines} lines excluding header)" 
                        else: 
                            return f"Records found: {total_lines}" 
                 
                # Look for DataFrame/table patterns  
                structure_patterns = [ 
                    r'dataframe shape:\s*\((\d+),\s*(\d+)\)', 
                    r'shape:\s*\((\d+),', 
                    r'(\d+)\s+rows' 
                ] 
                 
                for pattern in structure_patterns: 
                    match = re.search(pattern, output_lower) 
                    if match: 
                        rows = int(match.group(1)) 
                        return f"Records found: {rows}" 
             
            # Fallback: general counting patterns 
            general_patterns = [ 
                r'total[^:]*:\s*(\d+)', 
                r'count[^:]*:\s*(\d+)', 
                r'found\s+(\d+)', 
                r'(\d+)\s+(?:items|entries|records)' 
            ] 
             
            for pattern in general_patterns: 
                match = re.search(pattern, output_lower) 
                if match: 
                    count = match.group(1) 
                    return f"Count: {count}"
         
        # For exploration goals - extract key findings 
        elif 'explor' in goal.id.lower(): 
            findings = []
            if 'column' in output_lower or 'header' in output_lower: 
                findings.append("Data structure analyzed")
            if 'encoding:' in output_lower: 
                encoding_match = re.search(r'encoding:\s*([^\n]+)', output_lower) 
                if encoding_match:
                    findings.append(f"Encoding: {encoding_match.group(1).strip()}")
            return "; ".join(findings) if findings else "Exploration completed" 
         
        # For preparation goals 
        elif 'prepar' in goal.id.lower(): 
            if any(indicator in output_lower for indicator in ['loaded', 'prepared', 'ready', 'processed']): 
                return "Data preparation completed" 
            return "Preparation phase completed" 
         
        # For validation goals - extract validation results 
        elif 'validat' in goal.id.lower(): 
            # Look for final counting results in validation 
            if 'number of lines:' in output_lower: 
                line_match = re.search(r'number of lines:\s*(\d+)', output_lower) 
                if line_match: 
                    total_lines = int(line_match.group(1)) 
                    # Apply autonomous reasoning about headers 
                    record_count = total_lines - 1 if total_lines > 1 else total_lines
                    return f"Validation complete: {record_count} records confirmed" 

            return "Validation completed"
        return None 

     
    '''
    def _synthesize_final_answer(self) -> str: 
        """Synthesize the final answer from all completed goals using autonomous reasoning""" 
        answers = []
        key_findings = [] 
         
        # Collect answers from completed goals 
        for goal in self.memory.goal_tree.values(): 
            if goal.status == "completed" and hasattr(goal, 'extracted_answer') and goal.extracted_answer: 
                answers.append(f"{goal.id}: {goal.extracted_answer}") 
                 
                # Look for record/count results autonomously 
                if goal.extracted_answer: 
                    # Extract numerical results from any counting-related answer 
                    if 'records found:' in goal.extracted_answer.lower(): 
                        count_match = re.search(r'records found:\s*(\d+)', goal.extracted_answer.lower()) 
                        if count_match: 
                            count = count_match.group(1)
                            key_findings.append(f"**FINAL ANSWER: {count} records found**") 
                     
                    elif 'validation complete:' in goal.extracted_answer.lower(): 
                        validation_match = re.search(r'validation complete:\s*(\d+)', goal.extracted_answer.lower()) 
                        if validation_match: 
                            count = validation_match.group(1) 
                            key_findings.append(f"**FINAL ANSWER: {count} records (validated)**")
         
        # Autonomous fallback: analyze session history for counting patterns 
        if not key_findings: 
            for record in self.session_history: 
                if record.get('success') and record.get('output'): 
                    output = record['output'].lower() 
                     
                    # Look for structured data counting patterns 
                    if 'number of lines:' in output: 
                        line_match = re.search(r'number of lines:\s*(\d+)', output) 
                        if line_match: 
                            total_lines = int(line_match.group(1)) 
                            # Autonomous reasoning: check if headers were mentioned earlier
                            has_headers = any('header' in r.get('output', '').lower() for r in self.session_history if r.get('success'))
                            if has_headers and total_lines > 1: 
                                record_count = total_lines - 1 
                                key_findings.append(f"**FINAL ANSWER: {record_count} records ({total_lines} lines minus header)**") 
                            else: 
                                key_findings.append(f"**FINAL ANSWER: {total_lines} records**") 
                            break 
                     
                    # Look for DataFrame patterns 
                    elif 'dataframe shape:' in output: 
                        shape_match = re.search(r'dataframe shape:\s*\((\d+),\s*(\d+)\)', output) 
                        if shape_match: 
                            rows = shape_match.group(1) 
                            key_findings.append(f"**FINAL ANSWER: {rows} records in the dataset**") 
                            break
         
        if key_findings: 
            return "\n".join(key_findings) 
        elif answers: 
            return "Key findings: " + "; ".join(answers) 
        else:
            return "Analysis completed but no clear final answer extracted" 
        '''
    
    def _synthesize_final_answer(self) -> Dict[str, Any]:
        """Synthesize the final answer from all completed goals using autonomous reasoning"""

        answers = []
        key_findings = []

        # Collect answers from completed goals
        for goal in self.memory.goal_tree.values():
            if goal.status == "completed" and getattr(goal, "extracted_answer", None):
                answers.append(f"{goal.id}: {goal.extracted_answer}")

                # Detect record/count results
                if "records found:" in goal.extracted_answer.lower():
                    count_match = re.search(r"records found:\s*(\d+)", goal.extracted_answer.lower())
                    if count_match:
                        count = count_match.group(1)
                        key_findings.append(f"{count} records found")

                elif "validation complete:" in goal.extracted_answer.lower():
                    validation_match = re.search(r"validation complete:\s*(\d+)", goal.extracted_answer.lower())
                    if validation_match:
                        count = validation_match.group(1)
                        key_findings.append(f"{count} records (validated)")

        # Fallback: analyze session history
        if not key_findings:
            for record in self.session_history:
                if record.get("success") and record.get("output"):
                    output = record["output"].lower()

                    if "number of lines:" in output:
                        line_match = re.search(r"number of lines:\s*(\d+)", output)
                        if line_match:
                            total_lines = int(line_match.group(1))
                            has_headers = any(
                                "header" in r.get("output", "").lower()
                                for r in self.session_history
                                if r.get("success")
                            )
                            if has_headers and total_lines > 1:
                                record_count = total_lines - 1
                                key_findings.append(f"{record_count} records ({total_lines} lines minus header)")
                            else:
                                key_findings.append(f"{total_lines} records")
                        break

                    elif "dataframe shape:" in output:
                        shape_match = re.search(r"dataframe shape:\s*\((\d+),\s*(\d+)\)", output)
                        if shape_match:
                            rows = shape_match.group(1)
                            key_findings.append(f"{rows} records in the dataset")
                        break

        # ✅ Always return a dict
        if key_findings:
            return {
                "final_answer": key_findings[0],   # best guess
                "all_answers": key_findings,
                "reasoning": "Extracted from completed goals and session history",
            }
        elif answers:
            return {
                "final_answer": answers[0],
                "all_answers": answers,
                "reasoning": "Collected answers from completed goals",
            }
        else:
            return {
                "final_answer": None ,
                "all_answers": [],
                "reasoning": "Analysis completed but no clear final answer extracted",
            }

    def solve_autonomously(self, user_goal: str) -> str: 
        """Enhanced autonomous solving with advanced capabilities"""
        print(f"\n🚀 ENHANCED AUTONOMOUS AGENT ACTIVATED") 
        print(f"📋 GOAL: {user_goal}")
        print("=" * 80) 
         
        self.session_history = [] 
        self.global_execution_state = ExecutionState()  # Reset state 
        overall_success = False 
         
        # Step 1: Advanced Goal Analysis & Decomposition 
        print("🧠 ADVANCED GOAL ANALYSIS...") 
        self.memory.goal_tree = self._adaptive_goal_decomposition(user_goal) 
         
        print(f"📊 GOAL BREAKDOWN ({len(self.memory.goal_tree)} goals):") 
        for goal in self.memory.goal_tree.values(): 
            deps_str = f" (deps: {goal.dependencies})" if goal.dependencies else "" 
            print(f"  🎯 {goal.id}: {goal.description}{deps_str}") 
         
        # Step 2: Enhanced Goal Execution 
        for overall_iteration in range(1, self.max_iterations + 1): 
            current_goal = self._get_next_goal() 
             
            if not current_goal: 
                all_completed = all(g.status == "completed" for g in self.memory.goal_tree.values()) 
                if all_completed: 
                    print("🎉 ALL GOALS COMPLETED!") 
                    overall_success = True 
                    break 
                else: 
                    # Check for blocked goals 
                    blocked_goals = [g for g in self.memory.goal_tree.values() if g.status == "pending"] 
                    if blocked_goals: 
                        print("⚠️ Some goals are blocked by dependencies") 
                        # Try to unblock by marking failed dependencies as completed 
                        for goal in blocked_goals: 
                            if goal.retry_count < goal.max_retries: 
                                current_goal = goal 
                                break 
                     
                    if not current_goal: 
                        print("⚠️ No more executable goals available") 
                        break 

            current_goal.status = "in_progress" 
            current_goal.retry_count += 1 

            print(f"\n🎯 EXECUTING: {current_goal.description}")
            print(f"🔄 ITERATION {overall_iteration}/{self.max_iterations} | Attempt {current_goal.retry_count}/{current_goal.max_retries}")
            print("=" * 60)

            # Execute goal with enhanced attempts 
            goal_success = False 
            for goal_iteration in range(1, 4):  # Max 3 attempts per goal execution 
                print(f"\n📝 Goal Attempt {goal_iteration}/3")
                 
                # Enhanced reasoning with strategy selection 
                reasoning_result = self._enhanced_autonomous_reasoning(current_goal, goal_iteration) 
                 
                print(f"🧠 REASONING: {reasoning_result['reasoning'][:200]}...") 
                print(f"📐 STRATEGY: {reasoning_result['strategy'][:150]}...") 
                 
                # Install packages if needed 
                if reasoning_result.get('packages_needed'): 
                    print(f"📦 INSTALLING: {reasoning_result['packages_needed']}") 
                    for package in reasoning_result['packages_needed']: 
                        if package not in self.memory.installed_packages: 
                            self._install_package_safely(package)

                if reasoning_result['code']:
                    print(f"\n💻 EXECUTING ENHANCED CODE:") 
                    print("-" * 40) 
                    code_lines = reasoning_result['code'].split('\n') 
                    for i, line in enumerate(code_lines[:50], 1): 
                        print(f"{i:2d} | {line}") 
                    if len(code_lines) > 50: 
                        print(f"... ({len(code_lines) - 50} more lines)") 
                    print("-" * 40)
                    # Execute with state tracking
                    output, error, success = self._execute_code_with_state_tracking(reasoning_result['code'], current_goal) 


                    if output:
                        print("\n📤 OUTPUT:")
                        lines = output.split('\n')
                        for line in lines[:45]:
                            if line.strip():
                                print(line)
                        if len(lines) > 45:
                            print(f"... ({len(lines) - 45} more lines)")

                    if error and not success:
                        print("\n❌ ERROR:") 
                        error_lines = error.split('\n') 
                        for line in error_lines[:7]:
                            if line.strip():
                                print(line)

                    # Record enhanced session 
                    session_record = { 
                        'iteration': overall_iteration, 
                        'goal_id': current_goal.id, 
                        'goal_attempt': goal_iteration, 
                        'reasoning': reasoning_result['reasoning'], 
                        'strategy': reasoning_result.get('strategy', 'unknown'), 
                        'code': reasoning_result['code'], 
                        'output': output, 
                        'error': error if not success else None, 
                        'success': success, 
                        'execution_state_snapshot': len(self.global_execution_state.variables) 
                    } 
                    self.session_history.append(session_record) 
                     
                    # Enhanced evaluation 
                    if success: 
                        evaluation = self._enhanced_goal_evaluation(current_goal) 
                        print(f"\n🎯 EVALUATION: {evaluation['reason']} (Confidence: {evaluation['confidence']}%)") 
                         
                        if evaluation['completed'] and evaluation['confidence'] > 70: 
                            current_goal.status = "completed" 
                            current_goal.result = output 
                            # Store extracted answer if available 
                            if evaluation.get('extracted_answer'): 
                                current_goal.extracted_answer = evaluation['extracted_answer'] 
                            goal_success = True 
                            print(f"✅ GOAL COMPLETED!") 
                            if evaluation.get('extracted_answer'): 
                                print(f"🎯 EXTRACTED: {evaluation['extracted_answer']}") 
                            break 
                        elif evaluation['confidence'] > 50: 
                            # Partial success - continue to next goal 
                            current_goal.status = "completed" 
                            current_goal.result = output 
                            if evaluation.get('extracted_answer'): 
                                current_goal.extracted_answer = evaluation['extracted_answer'] 
                            goal_success = True 
                            print(f"✅ GOAL PARTIALLY COMPLETED (sufficient progress)!") 
                            if evaluation.get('extracted_answer'): 
                                print(f"🎯 EXTRACTED: {evaluation['extracted_answer']}") 
                            break 
                
                else:
                    print("⚠️ No code generated")
                # Add delay between attempts to prevent rapid failures 
                if goal_iteration < 3 and not goal_success: 
                    time.sleep(1) 
             
            if not goal_success: 
                current_goal.status = "failed" 
                print(f"❌ GOAL FAILED after {current_goal.retry_count} total attempts")
                # Try to continue with other goals even if one fails 
                continue 
         
        # Enhanced final evaluation 
        completed_count = sum(1 for g in self.memory.goal_tree.values() if g.status == "completed") 
        total_count = len(self.memory.goal_tree) 
        overall_success = completed_count > 0 
         
        print(f"\n📊 FINAL STATUS: {completed_count}/{total_count} goals completed") 
        print(f"🗄️ EXECUTION STATE: {self._summarize_execution_state()}") 

         
        # Synthesize and display final answer
        final_answer = self._synthesize_final_answer() 
        if final_answer and "FINAL ANSWER:" in final_answer: 
            print(f"\n🎉 {final_answer}")
        return {
            "reasoning": [r.get("reasoning") for r in self.session_history if r.get("reasoning")],
            "code": [r.get("code") for r in self.session_history if r.get("code")],
            "final_answer": final_answer,
            "overall_success": overall_success,
            "completed_goals": sum(1 for g in self.memory.goal_tree.values() if g.status == "completed"),
            "total_goals": len(self.memory.goal_tree),
        }




    def _format_enhanced_output(self, user_goal: str, overall_success: bool) -> str: 
        """Generate enhanced professional output with final answer synthesis"""
        # Get successful outputs 
        outputs = []
        strategies_used = set() 
        for record in self.session_history: 
            if record.get('success') and record.get('output'): 
                outputs.append(record['output']) 
                if record.get('strategy'): 
                    strategies_used.add(record['strategy']) 
         
        # Synthesize final answer 
        final_answer = self._synthesize_final_answer() 
         
        # Create enhanced report
        lines = []
        lines.append("╔" + "═" * 78 + "╗")
        lines.append("║" + " ENHANCED AUTONOMOUS AGENT EXECUTION REPORT".center(78) + "║")
        lines.append("╠" + "═" * 78 + "╣")
        status_icon = "✅" if overall_success else "⚠️" 
        status_text = "COMPLETED" if overall_success else "ATTEMPTED" 
        lines.append(f"║ {status_icon} STATUS: {status_text}") 
        lines.append(f"║ 🎯 GOAL: {user_goal[:60]}") 
        if len(user_goal) > 60:
            lines.append(f"║       {user_goal[60:120]}") 
        lines.append("║")
        lines.append("╠" + "═" * 78 + "╣")
        lines.append("║" + " FINAL ANSWER".center(78) + "║")
        lines.append("╠" + "═" * 78 + "╣") 
         
        if final_answer and "FINAL ANSWER:" in final_answer: 
            answer_line = final_answer.replace("**FINAL ANSWER:", "").replace("**", "").strip() 
            lines.append(f"║ 🎉 {answer_line}") 
        else: 
            # Fallback - try to extract from latest successful output 
            if outputs: 
                latest_output = outputs[-1] 
                shape_match = re.search(r'DataFrame shape:\s*\((\d+),\s*(\d+)\)', latest_output) 
                if shape_match: 
                    rows = shape_match.group(1) 
                    lines.append(f"║ 🎉 ANSWER: {rows} characters found in the dataset") 
                else:
                    lines.append("║ ⚠️ Analysis completed but final answer needs clarification") 
            else: 
                lines.append("║ ❌ No clear final answer generated") 
         
        lines.append("║") 
        lines.append("╠" + "═" * 78 + "╣") 
        lines.append("║" + " DETAILED RESULTS".center(78) + "║") 
        lines.append("╠" + "═" * 78 + "╣") 
         
        if outputs:
            result_text = outputs[-1]  # Latest output 
            result_lines = result_text.split('\n') 
            for line in result_lines[:10]: 
                if line.strip(): 
                    clean_line = line[:76].replace('║', '|')  # Prevent formatting issues 
                    lines.append(f"║ {clean_line}") 
            if len(result_lines) > 10: 
                lines.append(f"║ ... ({len(result_lines) - 10} more lines)") 
        else: 
            lines.append("║ No detailed results generated") 
         
        lines.append("║") 
        lines.append("╠" + "═" * 78 + "╣") 
        lines.append("║" + " EXECUTION ANALYTICS".center(78) + "║") 
        lines.append("╠" + "═" * 78 + "╣") 
        lines.append("║ 🔄 Total Iterations: " + str(len(self.session_history))) 
        lines.append("║ ✅ Successful Executions: " + str(sum(1 for r in self.session_history if r.get('success')))) 
        lines.append("║ 🧠 LLM Backend: " + str(self.active_llm)) 
        lines.append("║ 📦 Packages Installed: " + str(len(self.memory.installed_packages))) 
        lines.append("║ 🎯 Goals Completed: " + str(sum(1 for g in self.memory.goal_tree.values() if g.status == "completed"))) 
        lines.append("║ 🗄️ Data Artifacts: " + str(len(self.global_execution_state.data_artifacts))) 
        lines.append("║ 🔧 Variables in Scope: " + str(len(self.global_execution_state.variables))) 

        if strategies_used:
            lines.append("║ 🧩 Strategies Used: " + ", ".join(list(strategies_used)[:3]))
        lines.append("╚" + "═" * 78 + "╝")
        return "\n".join(lines)
     
    def get_enhanced_learned_info(self) -> Dict[str, Any]: 
        """Get comprehensive learned information""" 
        return { 
            "installed_packages": list(self.memory.installed_packages), 
            "context_discovered": dict(self.memory.context_discovered), 
            "active_llm": self.active_llm, 
            "execution_count": len(self.session_history), 
            "data_artifacts": dict(self.global_execution_state.data_artifacts), 
            "available_variables": list(self.global_execution_state.variables.keys()), 
            "exploration_results": dict(self.global_execution_state.exploration_results), 
            "goal_completion_rate": sum(1 for g in self.memory.goal_tree.values() if g.status == "completed") / len(self.memory.goal_tree) if self.memory.goal_tree else 0, 
            "success_rate": sum(1 for r in self.session_history if r.get('success')) / len(self.session_history) if self.session_history else 0 
        }

 

def main():
    """Enhanced main interface""" 
    print("🚀 Initializing Enhanced Truly Autonomous Agent...") 
    print("🧠 Advanced Chain-of-Thought & Complex Query Handling enabled!") 
    print("🔗 LLM Fallback: Gemini-1.5-pro") 
    print("✨ Enhanced Features: Hierarchical decomposition, State tracking, Adaptive reasoning") 

    try: 
        agent = TrulyAutonomousAgent()
        print("\n🤖 **ENHANCED TRULY AUTONOMOUS AGENT** Ready!") 
        print("✨ Advanced Capabilities:") 
        print("  • 🧠 Multi-strategy Chain-of-Thought reasoning") 
        print("  • 🎯 Hierarchical goal decomposition") 
        print("  • 🔗 3-Tier LLM fallback with enhanced error handling") 
        print("  • 📦 Intelligent package installation") 
        print("  • 🗄️ Advanced execution state tracking") 
        print("  • 🔄 Adaptive learning from failures") 
        print("  • 📊 Complex query analysis and handling") 
        print("\nType 'quit' to exit, 'learned' to see discoveries, 'state' for execution state.\n") 

        while True: 
            user_input = input("\n🎯 **Your Goal**: ").strip()

            if user_input.lower() in ['quit', 'exit', 'bye']: 
                print("\n👋 Goodbye!") 
                break 


            if user_input.lower() == 'learned': 
                learned_info = agent.get_enhanced_learned_info() 
                print("\n📚 COMPREHENSIVE LEARNED INFORMATION:") 
                for key, value in learned_info.items(): 
                    if isinstance(value, (list, dict)) and len(str(value)) > 100: 
                        print(f"  • {key}: {type(value).__name__} with {len(value)} items") 
                    else: 
                        print(f"  • {key}: {value}") 
                continue 


            if user_input.lower() == 'state': 
                print("\n🗄️ CURRENT EXECUTION STATE:") 
                print(f"  • Variables: {len(agent.global_execution_state.variables)}") 
                print(f"  • Data artifacts: {len(agent.global_execution_state.data_artifacts)}") 
                print(f"  • Functions: {len(agent.global_execution_state.functions)}") 
                print(f"  • Exploration results: {len(agent.global_execution_state.exploration_results)}") 
                if agent.global_execution_state.data_artifacts: 
                    print("  • Available data:") 
                    for name, info in list(agent.global_execution_state.data_artifacts.items())[:3]: 
                        print(f"    - {name}: {info.get('type', 'unknown')} {info.get('shape', '')}") 
                continue 

            if not user_input: 
                continue
            print("\n" + "🚀 " * 20)             

            try: 
                result = agent.solve_autonomously(user_input) 
                print(result) 

            except KeyboardInterrupt: 
                print("\n\n⏹️ Execution interrupted by user") 
            except Exception as e: 
                print(f"\n❌ System error: {str(e)}") 
                traceback.print_exc()
            print("\n" + "=" * 80)

    except Exception as e: 
        print(f"❌ Agent initialization failed: {e}") 
        traceback.print_exc()


if __name__ == "__main__":
    main()