#!/usr/bin/env python3
"""
Feedback Agent for summarizing assessment results
"""

import csv
import os
import logging
from typing import Dict, List, Optional
from collections import defaultdict
from pydantic import BaseModel
from agant_updated_assement import ProfileStorage, AssessmentItem, StudentProfile, TeacherProfile, ConfigurableLLM, get_config, AgentConfig
import json
import argparse
from statistics import mean, median

# Logging setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class FeedbackSummary(BaseModel):
    student_summaries: Dict[str, Dict] = {}
    teacher_summaries: Dict[str, Dict] = {}
    objective_summaries: Dict[str, Dict] = {}

class FeedbackAgent:
    def __init__(self, config: AgentConfig, profile_storage: ProfileStorage, llm: ConfigurableLLM, csv_path: str = "assessment_data.csv"):
        self.config = config
        self.profile_storage = profile_storage
        self.llm = llm
        self.csv_path = csv_path
        self.assessments: List[Dict] = []
        # Create CSV file with embedded data
        self._create_csv_file()
        logger.info(f"Initialized FeedbackAgent with CSV path: {csv_path}")

    def _create_csv_file(self):
        """Create a CSV file with embedded data."""
        csv_data = """student_id,teacher_id,objective,question,answer
student_001,teacher_001,Understand AI concepts,What is the primary role of ECU diagnostics in vehicles?,ECU diagnostics monitor vehicle systems to detect faults and ensure performance.
student_002,teacher_002,Apply AI concepts,How do AUTOSAR protocols enhance ECU data processing?,AUTOSAR's protocols enable consistent data exchange, improving ECU data processing efficiency.
student_003,teacher_003,Analyze AI applications,What advantages does AI bring to AUTOSAR-based fault detection?,AI analyzes AUTOSAR data to predict faults with higher accuracy using real-time ECU inputs.
student_004,teacher_004,Understand AI concepts,Why are ECU diagnostics critical for automotive safety?,ECU diagnostics ensure vehicle reliability by identifying system faults early.
student_005,teacher_005,Apply AI concepts,Describe a way AUTOSAR improves diagnostic data consistency.,AUTOSAR provides uniform data formats, enhancing diagnostic data consistency across ECUs.
student_006,teacher_006,Analyze AI applications,How can AI fault prediction benefit from AUTOSAR interfaces?,AI fault prediction benefits from AUTOSAR interfaces by accessing standardized ECU data for analysis.
student_007,teacher_001,Understand AI concepts,What is the main purpose of ECU diagnostics in automotive systems?,ECU diagnostics identify faults to maintain vehicle performance and safety.
student_008,teacher_002,Apply AI concepts,Explain how AUTOSAR supports efficient ECU troubleshooting.,AUTOSAR ensures reliable data exchange, supporting efficient ECU troubleshooting processes.
student_009,teacher_003,Analyze AI applications,What role does AI play in enhancing AUTOSAR diagnostics?,AI enhances AUTOSAR diagnostics by processing ECU data for predictive maintenance.
student_010,teacher_004,Understand AI concepts,How does ECU diagnostics contribute to vehicle reliability?,ECU diagnostics detect issues to ensure consistent vehicle reliability.
student_011,teacher_005,Apply AI concepts,What is a key benefit of using AUTOSAR for ECU diagnostics?,AUTOSAR's standardized protocols streamline ECU diagnostic data handling.
student_012,teacher_006,Analyze AI applications,How can AI integrate with AUTOSAR to predict ECU failures?,AI integrates with AUTOSAR by analyzing CAN bus data to predict ECU failures accurately."""
        
        with open(self.csv_path, 'w', newline='') as csvfile:
            csvfile.write(csv_data)

    def infer_bloom_level(self, question: str, answer: str) -> str:
        """Infer Bloom's Taxonomy level for a question and answer."""
        bloom_level = self.llm.assess_bloom_taxonomy(f"Question: {question}\nAnswer: {answer}")
        logger.debug(f"Inferred Bloom level: {bloom_level} for question: {question[:50]}... and answer: {answer[:50]}...")
        return bloom_level

    def infer_question_type(self, question: str, answer: str) -> str:
        """Infer question type based on question and answer characteristics."""
        word_count = len(answer.split())
        if word_count <= self.config.SHORT_ANSWER_MAX_WORDS:
            return "short_answer"
        elif word_count <= 200:
            return "open_ended"
        elif self.config.DESCRIPTIVE_MIN_WORDS <= word_count <= self.config.DESCRIPTIVE_MAX_WORDS:
            return "descriptive"
        return "open_ended"  # Default fallback

    def compute_score(self, question: str, answer: str, objective: str) -> float:
        """Compute a score for the answer using LLM-based relevance checking."""
        prompt = f"""
        Assess the student answer for relevance and correctness based on the question and content domain (AI in automotive systems, focusing on AUTOSAR, ECU diagnostics, or fault prediction).
        Question: {question}
        Student Answer: {answer}
        Objective: {objective}
        Return a JSON object with: {{ "score": float, "feedback": str }}
        Score from 0.0 to 1.0 based on accuracy and relevance.
        Award 1.0 for highly relevant and fully correct answers, 0.7-0.9 for partially correct or well-phrased answers with minor omissions, 0.5-0.6 for minimally relevant but related answers, 0.0-0.4 for incorrect or irrelevant answers.
        Be lenient with detailed answers that address the core concept, even with slight rephrasing or synonyms. Prioritize alignment with the objective over exact wording.
        Ensure the response is valid JSON with no additional text or markdown.
        """
        try:
            response = self.llm.invoke(prompt, max_tokens=500, temperature=0.2)
            result = json.loads(response)
            score = float(result['score'])
            logger.debug(f"LLM score: {score}, Feedback: {result['feedback']} for question: {question[:50]}... and answer: {answer[:50]}...")
            return max(0.0, min(1.0, score))
        except Exception as e:
            logger.warning(f"Error computing score with LLM: {e}. Falling back to keyword-based scoring.")
            automotive_keywords = ['ecu', 'autosar', 'diagnostics', 'fault', 'prediction', 'interface', 'data', 'monitor', 'safety', 'reliability', 'standardized']
            question_words = set(question.lower().split())
            answer_words = set(answer.lower().split())
            common_words = len(question_words & answer_words & set(automotive_keywords)) / len(question_words) if question_words else 0.0
            score = max(0.0, min(1.0, common_words * 0.8))
            logger.debug(f"Fallback score: {score} for question: {question[:50]}... and answer: {answer[:50]}...")
            return score

    def read_csv(self):
        """Read assessment data from CSV and update profiles with inferred attributes."""
        self.assessments = []
        try:
            with open(self.csv_path, 'r', newline='') as csvfile:
                reader = csv.DictReader(csvfile)
                expected_fields = {'student_id', 'teacher_id', 'objective', 'question', 'answer'}
                if not all(field in reader.fieldnames for field in expected_fields):
                    raise ValueError(f"CSV must contain fields: {', '.join(expected_fields)}")
                logger.info(f"Reading CSV from: {os.path.abspath(self.csv_path)}")

                for row in reader:
                    student_id = row['student_id']
                    teacher_id = row['teacher_id']
                    objective = row['objective']
                    question = row['question']
                    answer = row['answer']

                    bloom_level = self.infer_bloom_level(question, answer)
                    question_type = self.infer_question_type(question, answer)
                    score = self.compute_score(question, answer, objective)

                    assessment = AssessmentItem(
                        question_type=question_type,
                        question_text=question,
                        student_answer=answer,
                        score=score,
                        bloom_level=bloom_level,
                        objective=objective,
                        curriculum_standard=objective
                    )
                    self.assessments.append({
                        'student_id': student_id,
                        'teacher_id': teacher_id,
                        'assessment': assessment
                    })

                    # Update student profile
                    student_profile = StudentProfile(student_id=student_id, name=f"Student_{student_id}")
                    student_profile.assessments_taken = [assessment]
                    student_profile.total_assessments = 1
                    student_profile.average_score = score
                    self.profile_storage.save_student_profile(student_profile)

                    # Update teacher profile
                    teacher_profile = self.profile_storage.get_teacher_profile(teacher_id)
                    if not teacher_profile:
                        teacher_profile = TeacherProfile(teacher_id=teacher_id, name=f"Teacher_{teacher_id}")
                    if student_id not in teacher_profile.students:
                        teacher_profile.students.append(student_id)
                    self.profile_storage.save_teacher_profile(teacher_profile)

            logger.info(f"Read {len(self.assessments)} assessments from {self.csv_path}")
        except FileNotFoundError:
            logger.error(f"CSV file not found: {self.csv_path}")
            raise
        except Exception as e:
            logger.error(f"Error reading CSV: {e}")
            raise

    def generate_feedback(self) -> FeedbackSummary:
        """Generate feedback summarizing results."""
        summary = FeedbackSummary()
        student_scores = defaultdict(list)
        teacher_scores = defaultdict(list)
        objective_scores = defaultdict(list)
        student_objective_scores = defaultdict(lambda: defaultdict(list))
        student_bloom_scores = defaultdict(lambda: defaultdict(list))

        # Aggregate data from current assessments only
        for entry in self.assessments:
            student_id = entry['student_id']
            teacher_id = entry['teacher_id']
            assessment = entry['assessment']
            score = assessment.score
            objective = assessment.objective
            bloom_level = assessment.bloom_level

            student_scores[student_id].append(score)
            teacher_scores[teacher_id].append(score)
            objective_scores[objective].append(score)
            student_objective_scores[student_id][objective].append(score)
            student_bloom_scores[student_id][bloom_level].append(score)

        # Summarize per student
        for student_id, scores in student_scores.items():
            profile = self.profile_storage.get_student_profile(student_id)
            if profile:
                obj_breakdown = {obj: mean(scores) if scores else 0.0 for obj, scores in student_objective_scores[student_id].items()}
                bloom_breakdown = {bloom: mean(scores) if scores else 0.0 for bloom, scores in student_bloom_scores[student_id].items()}
                trend = "Stable"
                if len(scores) > 1 and scores[-1] > scores[0]:
                    trend = "Improving"
                elif len(scores) > 1 and scores[-1] < scores[0]:
                    trend = "Declining"
                summary.student_summaries[student_id] = {
                    'name': profile.name,
                    'average_score': mean(scores) if scores else 0.0,
                    'total_assessments': len(scores),
                    'objective_breakdown': obj_breakdown,
                    'bloom_breakdown': bloom_breakdown,
                    'performance_trend': trend,
                    'assessments': [{
                        'question': a.question_text,
                        'answer': a.student_answer,
                        'score': a.score,
                        'bloom_level': a.bloom_level,
                        'question_type': a.question_type,
                        'objective': a.objective
                    } for a in profile.assessments_taken]
                }

        # Summarize per teacher
        for teacher_id, scores in teacher_scores.items():
            teacher_profile = self.profile_storage.get_teacher_profile(teacher_id)
            if teacher_profile:
                all_assessments = []
                student_scores_list = []
                for sid in teacher_profile.students:
                    s_profile = self.profile_storage.get_student_profile(sid)
                    if s_profile:
                        all_assessments.extend(s_profile.assessments_taken)
                        student_scores_list.append(mean(a.score for a in s_profile.assessments_taken if a.score is not None) if s_profile.assessments_taken else 0.0)
                summary.teacher_summaries[teacher_id] = {
                    'teacher_name': teacher_profile.name,
                    'average_score': mean(scores) if scores else 0.0,
                    'total_students': len(teacher_profile.students),
                    'score_distribution': {
                        'min': min(student_scores_list) if student_scores_list else 0.0,
                        'max': max(student_scores_list) if student_scores_list else 0.0,
                        'median': median(student_scores_list) if student_scores_list else 0.0,
                        'average': mean(student_scores_list) if student_scores_list else 0.0
                    },
                    'top_students': sorted(teacher_profile.students, key=lambda sid: mean(a.score for a in self.profile_storage.get_student_profile(sid).assessments_taken if a.score is not None) if self.profile_storage.get_student_profile(sid).assessments_taken else 0.0, reverse=True)[:2],
                    'low_students': sorted(teacher_profile.students, key=lambda sid: mean(a.score for a in self.profile_storage.get_student_profile(sid).assessments_taken if a.score is not None) if self.profile_storage.get_student_profile(sid).assessments_taken else 0.0)[:2],
                    'summary': f"Teacher performance summary for {len(teacher_profile.students)} students with an average score of {mean(scores):.2f}."
                }

        # Summarize per objective
        total_students = len(set(entry['student_id'] for entry in self.assessments))
        for objective, scores in objective_scores.items():
            avg_score = mean(scores) if scores else 0.0
            passing_students = sum(1 for sid in student_scores.keys() if student_objective_scores[sid][objective] and mean(student_objective_scores[sid][objective]) >= 0.5)
            mastery_percentage = (passing_students / total_students * 100) if total_students else 0.0
            summary.objective_summaries[objective] = {
                'average_score': avg_score,
                'total_questions': len(scores),
                'mastery_percentage': mastery_percentage,
                'summary': f"Performance on {objective} with {len(scores)} questions, {mastery_percentage:.1f}% mastery."
            }

        logger.info("Generated feedback summary")
        return summary

    def print_feedback(self, feedback: FeedbackSummary):
        """Print the feedback summary in a formatted way."""
        print("\n" + "="*60)
        print("FEEDBACK REPORT")
        print("="*60)

        print("\nStudent Summaries:")
        print("-"*40)
        for student_id, summary_data in feedback.student_summaries.items():
            print(f"Student ID: {student_id}")
            print(f"Name: {summary_data['name']}")
            print(f"Average Score: {summary_data['average_score']:.2f}")
            print(f"Total Assessments: {summary_data['total_assessments']}")
            print(f"Performance Trend: {summary_data['performance_trend']}")
            print(f"Objective Breakdown: {', '.join(f'{obj}: {score:.2f}' for obj, score in summary_data['objective_breakdown'].items())}")
            print(f"Bloom Breakdown: {', '.join(f'{bloom}: {score:.2f}' for bloom, score in summary_data['bloom_breakdown'].items())}")
            print("Assessments:")
            for assessment in summary_data['assessments']:
                print(f"  Question: {assessment['question']}")
                print(f"  Answer: {assessment['answer']}")
                print(f"  Score: {assessment['score']:.2f}")
                print(f"  Bloom Level: {assessment['bloom_level']}")
                print(f"  Question Type: {assessment['question_type']}")
                print(f"  Objective: {assessment['objective']}")
            print("-"*40)

        print("\nTeacher Summaries (Teacher Performance):")
        print("-"*40)
        for teacher_id, summary_data in feedback.teacher_summaries.items():
            print(f"Teacher ID: {teacher_id}")
            print(f"Teacher Name: {summary_data['teacher_name']}")
            print(f"Average Score: {summary_data['average_score']:.2f}")
            print(f"Total Students: {summary_data['total_students']}")
            print(f"Score Distribution: Min={summary_data['score_distribution']['min']:.2f}, Max={summary_data['score_distribution']['max']:.2f}, Median={summary_data['score_distribution']['median']:.2f}, Avg={summary_data['score_distribution']['average']:.2f}")
            print(f"Top Students: {', '.join(summary_data['top_students']) or 'None'}")
            print(f"Low Students: {', '.join(summary_data['low_students']) or 'None'}")
            print(f"Summary: {summary_data['summary']}")
            print("-"*40)

        print("\nObjective Summaries:")
        print("-"*40)
        for objective, summary_data in feedback.objective_summaries.items():
            print(f"Objective: {objective}")
            print(f"Average Score: {summary_data['average_score']:.2f}")
            print(f"Total Questions: {summary_data['total_questions']}")
            print(f"Mastery Percentage: {summary_data['mastery_percentage']:.1f}%")
            print(f"Summary: {summary_data['summary']}")
            print("-"*40)

def main():
    print("Feedback Agent Demo")
    print("="*45)

    # Set up command-line argument parser
    parser = argparse.ArgumentParser(description="Feedback Agent Demo")
    parser.add_argument("--csv-path", type=str, default='assessment_data.csv', help="Path to CSV file")
    args = parser.parse_args()

    config = get_config()
    profile_storage = ProfileStorage(db_path=config.PROFILE_DB_PATH)
    llm = ConfigurableLLM(
        config=config,
        api_key=config.GROQ_API_KEY,
        model=config.LLM_MODEL,
        base_url=config.LLM_BASE_URL,
        max_tokens=config.MAX_TOKENS
    )
    feedback_agent = FeedbackAgent(config=config, profile_storage=profile_storage, llm=llm, csv_path=args.csv_path)

    try:
        # Read CSV and update profiles
        feedback_agent.read_csv()

        # Generate and print feedback
        feedback = feedback_agent.generate_feedback()
        feedback_agent.print_feedback(feedback)

    except Exception as e:
        logger.error(f"Error running FeedbackAgent: {e}")
        print(f"Error: {e}")

if __name__ == "__main__":
    main()