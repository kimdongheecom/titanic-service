from app.domain.service.titanic_service import TitanicService
from icecream import ic # type: ignore

# print(f'결정트리 활용한 검증 정확도 {None}')
# print(f'랜덤포레스트 활용한 검증 정확도 {None}')
# print(f'나이브베이즈 활용한 검증 정확도 {None}')
# print(f'KNN 활용한 검증 정확도 {None}')
# print(f'SVM 활용한 검증 정확도 {None}')


# 컨트롤러는 데이터 전처리, 학습, 평가, 배포 등의 작업을 수행한다.
class TitanicController:
    service = TitanicService()
    # 데이터 전처리 수행
    def preprocess(self, train, test):
        return self.service.preprocess(train, test)
    
    # 학습 수행
    def learning(self):
        return self.service.learning()
    # 평가 수행
    def evaluation(self):
        return self.service.evaluation()
    # 제출 수행
    def submit(self):
        return self.service.submit()
    # K-Fold 교차 검증 수행
    def create_k_fold(self):
        return self.service.create_k_fold()
    
    