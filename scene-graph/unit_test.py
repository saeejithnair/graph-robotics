if __name__ == "__main__":
    init_gemini()
    sample_folder = '/home/qasim/Projects/graph-robotics/scene-graph/perception/sample_images'
    result_folder = '/home/qasim/Projects/graph-robotics/scene-graph/perception/sample_results'
    img_resize = (768,432) 
    device='cuda:0'
    sam_model_type = "vit_h"
    sam_checkpoint_path = "checkpoints/sam_vit_h_4b8939.pth"
    perceptor = Perceptor()
    for file in os.listdir(sample_folder):
        img = open_image(os.path.join(sample_folder, file))
        perceptor.process(img)
        perceptor.save_results(
            os.path.join(result_folder, file)[:-4] + '.txt', 
            os.path.join(result_folder, file))