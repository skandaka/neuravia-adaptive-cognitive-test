import json
import asyncio
from Data_generation.question_bank_generator import QuestionBankGenerator
from adaptive_selector import AdaptiveTestingEngine
from rgat_network import RGAT, CognitiveGraphBuilder
from Data_generation.openai_integration import OpenAIQuestionGenerator
import torch

def main():
    """Main execution function"""
    
    print("=== NeuraVia Adaptive Cognitive Test System ===\n")
    
    # 1. Generate Question Bank
    print("1. Generating 1,500 question bank...")
    generator = QuestionBankGenerator()
    question_bank = generator.generate_full_bank()
    generator.export_to_json('data/questions.json')
    
    stats = generator.get_statistics()
    print(f"   Generated {stats['total']} total questions")
    for module in ['concentration', 'calculation', 'simulation']:
        print(f"   - {module}: {sum(stats[module].values())} questions")
    
    # 2. Initialize Adaptive Selector
    print("\n2. Initializing adaptive selection engine...")
    adaptive_engine = AdaptiveTestingEngine(window_size=3)
    print("   Adaptive engine ready")
    
    # 3. Initialize R-GAT Network
    print("\n3. Building R-GAT neural network...")
    rgat = RGAT(in_features=4, hidden_features=64, out_features=32)
    graph_builder = CognitiveGraphBuilder()
    
    # Example cognitive scores
    sample_scores = {
        'concentration': [0.7, 0.8, 0.6, 0.75],
        'calculation': [0.6, 0.7, 0.8, 0.65],
        'simulation': [0.8, 0.7, 0.7, 0.8]
    }
    
    node_features, edge_index, edge_type = graph_builder.build_graph(sample_scores)
    
    # Forward pass through R-GAT
    with torch.no_grad():
        embeddings = rgat(node_features, edge_index, edge_type)
    
    print(f"   R-GAT output shape: {embeddings.shape}")
    print("   R-GAT network ready")
    
    # 4. Demonstrate adaptive selection
    print("\n4. Demonstrating adaptive question selection...")
    
    # Simulate a test session
    current_difficulty = 1
    responses = []
    
    for i in range(5):
        # Select next question
        question, next_difficulty = adaptive_engine.select_next_question(
            current_difficulty,
            responses,
            question_bank['concentration']
        )
        
        if question:
            print(f"   Question {i+1}: Difficulty {next_difficulty}")
            
            # Simulate response
            response = {
                'question_id': question['id'],
                'correct': i % 2 == 0,  # Alternate correct/incorrect for demo
                'time': 25 + i * 5,
                'difficulty': next_difficulty
            }
            responses.append(response)
            adaptive_engine.update_performance_model(response)
            
            current_difficulty = next_difficulty
    
    # Get performance summary
    summary = adaptive_engine.get_performance_summary()
    print(f"\n   Performance Summary:")
    print(f"   - Total Questions: {summary['total_questions']}")
    print(f"   - Accuracy: {summary['overall_accuracy']:.1%}")
    print(f"   - Avg Difficulty: {summary['average_difficulty']:.1f}")
    
    print("\n=== System Ready for Integration ===")
    print("All components successfully initialized and tested!")

if __name__ == "__main__":
    main()
