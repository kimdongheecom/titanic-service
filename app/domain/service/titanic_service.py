import numpy as np
from app.domain.model.data_schema import DataSchema        # 자료는 데이터셋에서 가져오겠다는 뜻이다. from "파일명", import "class"
import pandas as pd # pd = Pandas  #pandas는 데이터 분석 및 처리하는 라이브러리를 의미함.

# 추가된 머신러닝 관련 import
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier



class TitanicService:

    dataschema = DataSchema()   #DataSchema()을 dataschema이라고 이 안에서 부르겠다는 의미이다.
    # 모델 생성
    def new_model(self,fname) -> object: #self는 자기 데이터를 가지고 오겠다라는 의미. #new_model은 모델을 만드는 것을 의미함. fname이라는 파일명을 받고 object에 결과를 찍겠다
        this = self.dataschema #self가 붙으면 property인 것을 알수 있다. 그리고 self.dataschema을 this라고 설정하였다.
        this.context = 'C:\\Users\\bitcamp\\Documents\\kpmg-250424\\kpmg2501\\V2\\ai-server\\titanic-service\\app\\domain\\stored_data\\' #this라는 데이터 셋에 'C:\\\\Users\\bitcamp\\OneDrive\\문서\\titanic\\com\\kimdonghee\\datas\\titanic\\\\'라는 경로를 주겠다는 것이다.
        this.fname = fname #파일명만 유일하게 바깥에서 가져올 수 있다.
        
        return pd.read_csv(this.context + this.fname) #데이터 수집
    
    # 모델 전처리
    def preprocess(self, train_fname, test_fname) -> object: #데이터 전처리
        print("---------모델 전처리 시작 ----------")
        
        feature = ['PassengerId', 'Survived', 'Pclass', 'Name', 'Sex', 'Age', 
                   'SibSp', 'Parch', 'Ticket', 'Fare', 'Cabin', 'Embarked']
        this = self.dataschema
        this.train = self.new_model(train_fname)
        print("트레인 데이터")
        #ic(this.train) #ic는 프린터하는 기능이다.
        print(this.train)
        this.test = self.new_model(test_fname)
        print("테스트 데이터")
        #ic(this.train)
        this.id = this.test['PassengerId'] #테스트는 중간고사, 기말고사이고, 트레인은 매일 쪽지 시험 느낌////
        #'SibSp', 'parch'. 'Cabin', 'Ticket'가 지워야 할 feature 이다.
        this.label = this.train['Survived']
        this.train = this.train.drop('Survived', axis = 1)
        drop_features = ['SibSp', 'Parch', 'Cabin', 'Ticket']
        this = self.extract_title_from_name(this)
        title_mapping = self.remove_duplicate_title(this)
        this = self.title_nominal(this, title_mapping)
        this = self.drop_feature(this,'Name')
        this = self.gender_nominal(this)
        this = self.drop_feature(this,'Sex')
        this = self.embarked_norminal(this)  
        # self.df_info(this)
        this = self.age_ratio(this)
        this = self.drop_feature(this,'Age')
        this = self.pclass_ordnal(this)
        this = self.fare_ordinal(this)
        this = self.drop_feature(this,"Fare")

        return this
    
    # K-Fold 교차 검증
    def create_k_fold(self):
        this = self.dataschema
        numeric_cols = ['Pclass', 'Gender', 'AgeGroup', 'FareGroup', 'Embarked']
        X = this.train[numeric_cols]
        y = this.label

        # 명시적으로 KFold 지정
        kf = KFold(n_splits=5, shuffle=True, random_state=0)

        model = RandomForestClassifier(random_state=42)
        scores = cross_val_score(model, X, y, cv=kf)  # 5-Fold
        print(f"📚 K-Fold 정확도 평균: {scores.mean():.4f}, 표준편차: {scores.std():.4f}")
        return scores

    # 랜덤 피처 추가
    def create_randome_variable(self):
        this = self.dataschema
        np.random.seed(42)
        this.train['RandomFeature'] = np.random.randn(len(this.train))
        this.test['RandomFeature'] = np.random.randn(len(this.test))
        print("🎲 랜덤 피처(RandomFeature) 추가 완료")
        return this
    
     # 💡 정확도 비교용 알고리즘
    # 결정트리 정확도 비교
    def accuracy_by_dtree(self):
        print("🌲 DecisionTreeClassifier")
        return self._train_and_evaluate(DecisionTreeClassifier())

    # 랜덤포레스트 정확도 비교
    def accuracy_by_random_forest(self):
        print("🌳 RandomForestClassifier")
        return self._train_and_evaluate(RandomForestClassifier())

    # 나이브베이즈 정확도 비교
    def accuracy_by_naive_bayes(self):
        print("🧠 NaiveBayes (GaussianNB)")
        return self._train_and_evaluate(GaussianNB())

    # KNN 정확도 비교
    def accuracy_by_knn(self):
        print("👣 K-Nearest Neighbors")
        return self._train_and_evaluate(KNeighborsClassifier())
    
    # SVM 정확도 비교
    def accuracy_by_svm(self):
        print("💫 Support Vector Machine")
        return self._train_and_evaluate(SVC())

    # 🔁 공통 학습 및 평가 로직
    def _train_and_evaluate(self, model):
        this = self.dataschema
        numeric_cols = ['Pclass', 'Gender', 'AgeGroup', 'FareGroup', 'Embarked']
        X = this.train[numeric_cols]
        y = this.label
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)
        acc = accuracy_score(y_val, y_pred)
        print(f"✅ 정확도: {acc:.4f}")
        return acc


    @staticmethod #staicmethod는 함수가 아니다.
    def create_labels(this) -> object:
        return this.train['Survived'] #this는 self.dataschema을 변화시키려고 만든 것이다.
    
    @staticmethod
    def create_train(this) -> object:
        return this.train.drop('Survived', axis= 1) #Survived 답을 제거해야하니까.....그리고 이 함수는 답인지 알려줘야하니까 썼다.
    
    @staticmethod
    def drop_feature(this, *feature) -> object: #피쳐를 통해서 베러블을 가공시키겠다.... *표시와 같이 있으면 두번 묶는다는 것을 의미함. 리스트를 아규먼트로 보내면, 두개로 묶어지는 거다.....
        
        [i. drop(j, axis=1) for j in feature for i in [this.train, this.test]]

        # for i in [this.train, this.test]:
        #     for j in feauture:
        #         i.drop(j, axis=1, inplace=True)
        return this
    
    @staticmethod
    def df_info(this):    #this는 인간이 보는 것, self는 기계가 보는 것
        return this

    @staticmethod
    def extract_title_from_name(this):
        # for i in [this.train, this.test]:
        #     i['Name'].str.extract('([A-Za-z]+)\\.', expand = False)

        [i.__setitem__('Title',i['Name'].str.extract('([A-Za-z]+)\\.', expand = False)) 
                      for i in [this.train, this.test]] 
        return this
    
    
    @staticmethod
    def remove_duplicate_title(this):
        
        a = []
        for i in [this.train,this.test]:
        # [i.__setitem__('Title_Set', set(i['Title'])) for i in [this.train,this.test]]
            a += list(set(i['Title'])) #train, test 두번을 누적해야해서
        #[a = set(i['Title']) for i in [this.train,this.test]]
        a =list(set(a)) #train, test 각각은 중복이 아니지만, 합치면서 중복 발생
        print("🚚🚚a", a)
        #['Lady', 'Mr', 'Don', 'Miss', 'Countess', 'Dona', 'Jonkheer', 'Master', 
        #'Dr', 'Mme', 'Major', 'Col', 'Mrs', 'Capt', 'Rev', 'Mlle', 'Sir', 'Ms']
        '''
        ['Mr', 'Sir', 'Major', 'Don', 'Rev', 'Countess', 'Lady', 'Jonkheer', 'Dr',
        'Miss', 'Col', 'Ms', 'Dona', 'Mlle', 'Mme', 'Mrs', 'Master', 'Capt']
        Royal : ['Countess', 'Lady', 'Sir']
        Rare : ['Capt','Col','Don','Dr','Major','Rev','Jonkheer','Dona','Mme' ]
        Mr : ['Mlle']
        Ms : ['Miss']
        Master
        Mrs
        '''
        title_mapping = {'Mr': 1, 'Ms': 2, 'Mrs': 3, 'Master': 4, 'Royal': 5, 'Rare': 6}
        return title_mapping

    @staticmethod
    def kwargs_sample(**kwargs) -> None:
        {print(''.join(f'키워드 arg: {i} 값: {j}')) for i, j in kwargs.items()}

    @staticmethod
    def null_check(this):
        [print(i.isnull().sum()) for i in [this.train, this.test]]

    @staticmethod
    def title_nominal(this, title_mapping):
        for i in [this.train, this.test]:
            i['Title'] = i['Title'].replace(['Countess', 'Lady', 'Sir'], 'Royal')
            i['Title'] = i['Title'].replace(['Capt','Col','Don','Dr','Major','Rev','Jonkheer','Dona','Mme'], 'Rare')
            i['Title'] = i['Title'].replace(['Mlle'], 'Mr')
            i['Title'] = i['Title'].replace(['Miss'], 'Ms')
            # Master 는 변화없음
            # Mrs 는 변화없음
            i['Title'] = i['Title'].fillna(0) #0은 평민을 의미하고, 빈칸이었던 사람들을 다 0으로 채웠다. 이 사람들은 1에서 6까지 포함되지 않은 사람들이다.
            i['Title'] = i['Title'].map(title_mapping)
        print("😡😷🤢title_nominal 메쏘드 성공")
            
        return this
    
    @staticmethod
    def pclass_ordnal(this):      
        return this

    @staticmethod
    def gender_nominal(this):
        gender_mapping = {'male':1, 'female':2}

        [i.__setitem__('Gender',i['Sex'].map(gender_mapping)) 
         for i in [this.train, this.test]]
        
        # for i in [this.train,this.test]:
        #     i["Gender"] = i["Sex"].map(gender_mapping)

        # this.train['Gender'] = this.train['Sex'].map(gender_mapping)
        # this.test['Gender'] = this.test['Sex'].map(gender_mapping)

        print("🥟🍗🍗gender_nominal 메쏘드 성공")

        return this

    @staticmethod
    def age_ratio(this):
        TitanicService.get_Count_age_null(this, "Age")
        for i in [this.train, this.test]:
            i["Age"] = i["Age"].fillna(-0.5)
        age_mapping = {'Unknown':0 , 'Baby': 1, 'Child': 2, 'Teenager' : 3, 'Student': 4,
                       'Young Adult': 5, 'Adult':6,  'Senior': 7}
        # for i in [this.train, this.test]:
        #     missing_count = i["Age"].isnull().sum()
        #     print("🚕🚓missing_count age 빈 갯수 찾기", missing_count)      
        TitanicService.get_Count_age_null(this, "Age")
        train_max_age = max(this.train['Age'])
        test_max_age = max(this.test['Age'])
        max_age = max(train_max_age,test_max_age)
        print("👮‍♀️🧔😶😪🙄최고령자",max_age)
        bins=[-1, 0, 5, 12, 18, 24, 35, 60, np.inf]
        labels = ['Unknown','Baby', 'Child', 'Teenager', 'Student', 'Young Adult', 'Adult', 'Senior']

        
        age_mapping = {'Unknown':0 , 'Baby': 1, 'Child': 2, 'Teenager' : 3, 'Student': 4,
                       'Young Adult': 5, 'Adult':6,  'Senior': 7}
        for i in [this.train, this.test]:
            i['AgeGroup'] = pd.cut(i['Age'],bins, labels = labels).map(age_mapping)
        print("🍜🥩🧵🎞----age_ratio 성공------")
        
        return this

    @staticmethod
    def get_Count_age_null(this, feature):

        # for i in [this.train, this.test]:
        #     null

        # for i in range(len(this.train['AgeGroup'])):
        #     if this.train['AgeGroup'][i] == 'unknown':
        #         this.train['AgeGroup'][i] == age_mapping[this.train['AgeGroup'][i]]
        

            

         #몇살부터 몇살까지 구간을 정할때!,  60 이상은 np.inf라고 정한다.   
        # for i in [this.train, this.test]:
        #     i["Age"].isnull().fillna()
        #     print("fillna(0)이후 Age가 빈값의 갯수", missing_count)

        # missing_count = this.test["Age"].isnull().sum()
        # i["Age"].isnull().sum()
        # this.train["Age"].isnull().fillna(0, inplace = True)
        # print("🚕🚓missing_count age 빈 갯수 찾기", missing_count)
        
        
        # this.train['AgeGroup'] = pd.cut(this.train['Age'], bins=[1, 12, 23, 34, 45, 56, 68, 80],
        # labels=['Baby', 'Child', 'Teenager', 'Student', 'Young Adult', 'Adult', 'Senior'], right=False).map({
        #     'Baby': 1,
        #     'Child': 2,
        #     'Teenager': 3,
        #     'Student': 4,
        #     'Young Adult': 5,
        #     'Adult': 6,
        #     'Senior': 7})
        return this

    @staticmethod
    def fare_ordinal(this):
        print("😎😪😫🙄----fare _ratio 처음--------")
        TitanicService.get_Count_age_null(this, "Fare")
        for i in [this.train, this.test]:
            i["Fare"] = i["Fare"].fillna(0.5) #fillna함수에 넣는 것은 bins구간 안에 있는 숫자 값을 집어 넣으면 된다.
        this.train = this.train.fillna({"FareBand":1})
        this.test = this.test.fillna({"FareBand":1})

        
        # fare_mapping = {'Unknown':0 , 'Baby': 1, 'Child': 2, 'Teenager' : 3, 'Student': 4,
        #                'Young Adult': 5, 'Adult':6,  'Senior': 7}
        # for i in [this.train, this.test]:
        #     missing_count = i["Age"].isnull().sum()
        #     print("🚕🚓missing_count age 빈 갯수 찾기", missing_count)      
        # TitanicService.get_Count_age_null(this, "Fare")
        train_max_fare = max(this.train['Fare'])
        test_max_fare = max(this.test['Fare'])
        max_fare = max(train_max_fare,test_max_fare)
        print("😎😪😫🙄----fare _ratio 끝--------")
        print("👮‍♀️🧔😶😪🙄비용",max_fare)
        bins=[0, 128.0823, 256.1646, 384.2469, 512.3292]
        labels = ['Low', 'Medium-Low', 'Medium-High', 'High']

        fare_mapping = {'Low': 1 , 'Medium-Low': 2, 'Medium-High': 3, 'High' : 4}
        
        for i in [this.train, this.test]:
            i['FareGroup'] = pd.cut(i['Fare'],bins, labels = labels).map(fare_mapping)
            print("🎉🎉✨✨:", i['FareGroup'])
        
        # this.train['FareGroup'] = pd.qcut(this.train['Fare'], 4, labels = {1,2,3,4})

        return this

    @staticmethod
    def embarked_norminal(this):
        #embarked = this.train.fillna({'Embarked': 's'}) #fill채워라 NaN을..... ['Embarked'] 형태는 키만 가지고 있는 상태를 의미함. #'S'는 S라는 항구가 가장 많이 탔다. 
        this.train = this.train.fillna({'Embarked': 'S'})
        this.test = this.test.fillna({'Embarked': 'S'})
        embarked_mapping = {'S':1, 'C':2, 'Q':3}
        this.train['Embarked'] = this.train['Embarked'].map(embarked_mapping)
        this.test['Embarked'] = this.test['Embarked'].map(embarked_mapping)
        print("👦👩‍🦰👨‍🦰👱‍♂️embarked_norminal 메쏘드 성공")
        return this

# 데이터 셋에서 만들었던 객체들을 경로를 가져온 클래스 하는 작업...
# reuse하는 작업 ..get unit count!!
# 파일을 불러오는 작업
# 객체를 형성하는 작업((메모리를 로딩시켜서 사람이 볼수있는 상태로 만들어주는 것이다.))
    
    
    @staticmethod
    def print_this(this):
        print('*' * 100)
        print(f'1. Train 의 type \n {type(this.train)} ')
        print(f'2. Train 의 column \n {this.train.columns} ')
        print(f'3. Train 의 상위 1개 행\n {this.train.head()} ')
        print(f'4. Train 의 null 의 갯수\n {this.train.isnull().sum()}개')
        print(f'5. Test 의 type \n {type(this.test)}')
        print(f'6. Test 의 column \n {this.test.columns}')
        print(f'7. Test 의 상위 1개 행\n {this.test.head()}개')
        print(f'8. Test 의 null 의 갯수\n {this.test.isnull().sum()}개')
        print('*' * 100)
    # 모델끼리 라우터 하는 역할;;;;;
    # 두개를 가져와서 안에서 왔다갔다 하는 것이다.
    # 컨트롤러는 출력하는 것이다. 
    # 프린터 하는 것은 다 컨트롤러에서 한다.

    # 머신러닝: learning
    def learning(self):
        print("----- 학습 시작 -----")
        this = self.dataschema
        
        # 문자열 특성 제외하고 수치형 특성만 사용
        numeric_cols = ['Pclass', 'Gender', 'AgeGroup', 'FareGroup', 'Embarked']
        X = this.train[numeric_cols]
        y = this.label
        
        # 랜덤 포레스트 분류기 모델 생성
        model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
        
        # 5-Fold 교차 검증으로 모델 평가
        scores = cross_val_score(model, X, y, cv=5)  # 5-Fold
        print(f"📚 K-Fold 정확도 평균: {scores.mean():.4f}, 표준편차: {scores.std():.4f}")
        
        # 최종 모델 학습
        model.fit(X, y)
        
        # 모델 저장
        this.model = model
        this.feature_columns = numeric_cols
        print("----- 학습 완료 -----")
        
        return this
    
    def evaluation(self):
        print("----- 평가 시작 -----")
        this = self.dataschema
        
        if not hasattr(this, 'model'):
            print("모델이 학습되지 않았습니다. 먼저 learning() 메소드를 호출하세요.")
            return None
        
        # 결정트리 평가
        dtree_acc = self.accuracy_by_dtree()
        
        # 랜덤 포레스트 평가
        rf_acc = self.accuracy_by_random_forest()
        
        # 나이브 베이즈 평가
        nb_acc = self.accuracy_by_naive_bayes()
        
        # KNN 평가
        knn_acc = self.accuracy_by_knn()
        
        # SVM 평가
        svm_acc = self.accuracy_by_svm()
        
        results = {
            "decision_tree": dtree_acc,
            "random_forest": rf_acc,
            "naive_bayes": nb_acc,
            "knn": knn_acc,
            "svm": svm_acc
        }
        
        print("----- 평가 완료 -----")
        return results
    
    def submit(self):
        print("----- 제출 시작 -----")
        this = self.dataschema
        
        if not hasattr(this, 'model'):
            print("모델이 학습되지 않았습니다. 먼저 learning() 메소드를 호출하세요.")
            return None
        
        # 테스트 데이터에 대한 예측
        X_test = this.test[this.feature_columns]
        
        # 예측 결과
        predictions = this.model.predict(X_test)
        
        # 제출 파일 생성
        submission = pd.DataFrame({
            'PassengerId': this.id,
            'Survived': predictions
        })
        
        # 제출 파일 저장
        submission_path = this.context + 'submission.csv'
        submission.to_csv(submission_path, index=False)
        
        print(f"제출 파일이 저장되었습니다: {submission_path}")
        print("----- 제출 완료 -----")
        
        return submission