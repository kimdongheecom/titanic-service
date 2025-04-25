from app.domain.controller.titanic_controller import TitanicController

def main():
    controller = TitanicController()
    
    # 데이터 전처리 수행
    train_fname = 'train.csv'
    test_fname = 'test.csv'
    result = controller.preprocess(train_fname, test_fname)
    
    print("😚☺🙂🤗전처리 완료", result)
    
    # K-Fold 교차 검증 수행
    k_fold_scores = controller.create_k_fold()
    
    print("😎🧐🤓K-Fold 교차 검증 결과:", k_fold_scores)
    
    # 학습 수행
    learning_result = controller.learning()
    print("🧠 학습 완료:", learning_result)
    
    # 평가 수행
    evaluation_result = controller.evaluation()
    print("📊 평가 완료:", evaluation_result)
    
    # 제출 수행
    submission_result = controller.submit()
    print("📤 제출 완료:", submission_result)

if __name__ == "__main__":
    main()
