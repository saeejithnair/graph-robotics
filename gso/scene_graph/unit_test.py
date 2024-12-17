from perception import GenericMapper

perceptor = GenericMapper()
perceptor.init()
perceptor.load_results(
    "/pub3/qasim/hm3d/data/ham-sg/000-hm3d-BFRyYbPCCPE/detections", 1
)
print(perceptor)
