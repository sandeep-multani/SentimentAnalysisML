using Microsoft.Data.DataView;
using Microsoft.ML;
using Microsoft.ML.Data;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;

namespace SentimentAnalysis
{
    class Program
    {
        static readonly string _dataPath = Path.Combine(Environment.CurrentDirectory, "Data", "yelp_labelled.txt");
        static readonly string _modelPath = Path.Combine(Environment.CurrentDirectory, "Data", "Model.zip");

        static void Main(string[] args)
        {
            //Create ML.NET context/environment 
            //It allows you to add steps in order to keep everything together during ML process.
            MLContext mlContext = new MLContext();

            //Load data for training
            TrainCatalogBase.TrainTestData splitDataView = LoadData(mlContext);

            //Build and train model
            ITransformer model = BuildAndTrainModel(mlContext, splitDataView.TrainSet);

            //Evaluate model using test data
            //optionally save it for future use
            Evaluate(mlContext, model, splitDataView.TestSet);

            //Use model with single data item
            UseModelWithSingleItem(mlContext, model);

            //Load saved model and use with multiple data items
            UseLoadedModelWithBatchItems(mlContext);

            //Load saved model and use with user input
            UseLoadedModelWithUserInput(mlContext);

            Console.WriteLine();
            Console.WriteLine("=== End of process ===");
            Console.WriteLine();
            Console.WriteLine("Press any key to continue...");
            Console.ReadKey();

        }

        public static TrainCatalogBase.TrainTestData LoadData(MLContext mlContext)
        {
            //Load training data from text file
            //Please note that you can also load data from databases or in-memory collections.
            IDataView dataView = mlContext.Data.LoadFromTextFile<SentimentData>(_dataPath, hasHeader: false);
            //Split data into training and test data
            //The testFraction 0.2 will use 80% data for training and 20% data for testing the model
            TrainCatalogBase.TrainTestData splitDataView = mlContext.BinaryClassification.TrainTestSplit(dataView, testFraction: 0.2);
            return splitDataView;
        }

        public static ITransformer BuildAndTrainModel(MLContext mlContext, IDataView splitTrainSet)
        {
            //Create a flexible pipeline, composed by a chain of estimators for creating and training the model
            var pipeline = mlContext.Transforms.Text.FeaturizeText(outputColumnName: DefaultColumnNames.Features, inputColumnName: nameof(SentimentData.SentimentText))
            //Adds a FastTreeBinaryClassificationTrainer, the decision tree learner for this example
                .Append(mlContext.BinaryClassification.Trainers.FastTree(numLeaves: 50, numTrees: 50, minDatapointsInLeaves: 20));

            //Create and train model
            Console.WriteLine("=== Create and train the model ====");
            var model = pipeline.Fit(splitTrainSet);
            Console.WriteLine("=== End of training ===");
            Console.WriteLine();
            return model;
        }

        public static void Evaluate(MLContext mlContext, ITransformer model, IDataView splitTestSet)
        {
            // Evaluate the model and show accuracy stats
            Console.WriteLine("=== Evaluating model accuracy with test data ===");
            //Take the data in, make transformations, output the data.
            IDataView predictions = model.Transform(splitTestSet);
            // BinaryClassificationContext.Evaluate returns computed overall metrics.
            CalibratedBinaryClassificationMetrics metrics = mlContext.BinaryClassification.Evaluate(predictions, "Label");

            Console.WriteLine();
            Console.WriteLine("Model quality metrics evaluation");
            Console.WriteLine("--------------------------------");
            Console.WriteLine($"Accuracy: {metrics.Accuracy:P2}");
            Console.WriteLine($"Auc: {metrics.Auc:P2}");
            Console.WriteLine($"F1Score: {metrics.F1Score:P2}");
            Console.WriteLine("=== End of model evaluation ===");

            // Save the new model to .ZIP file
            SaveModelAsFile(mlContext, model);
        }

        private static void UseModelWithSingleItem(MLContext mlContext, ITransformer model)
        {
            //user model with single data time
            PredictionEngine<SentimentData, SentimentPrediction> predictionFunction = model.CreatePredictionEngine<SentimentData, SentimentPrediction>(mlContext);
            SentimentData sampleStatement = new SentimentData
            {
                SentimentText = "This was a very bad steak"
            };
            var resultprediction = predictionFunction.Predict(sampleStatement);
            Console.WriteLine();
            Console.WriteLine("=== Prediction test of model with a single sample and test dataset ===");
            Console.WriteLine();
            Console.WriteLine($"Sentiment: {sampleStatement.SentimentText} | Prediction: {(Convert.ToBoolean(resultprediction.Prediction) ? "Positive" : "Negative")} | Probability: {resultprediction.Probability} ");
            Console.WriteLine("=== End of predictions ===");
            Console.WriteLine();
        }

        public static void UseLoadedModelWithBatchItems(MLContext mlContext)
        {
            //Use model with a bacth of data items
            IEnumerable<SentimentData> sentiments = new[]
            {
                new SentimentData
                {
                    SentimentText = "This was a horrible meal"
                },
                new SentimentData
                {
                    SentimentText = "I love this spaghetti."
                }
            };

            //Load the saved (already trained) model
            ITransformer loadedModel;
            using (var stream = new FileStream(_modelPath, FileMode.Open, FileAccess.Read, FileShare.Read))
            {
                loadedModel = mlContext.Model.Load(stream);
            }

            //Load batch of data items and make predictions
            IDataView sentimentStreamingDataView = mlContext.Data.LoadFromEnumerable(sentiments);
            IDataView predictions = loadedModel.Transform(sentimentStreamingDataView);
            IEnumerable<SentimentPrediction> predictedResults = mlContext.Data.CreateEnumerable<SentimentPrediction>(predictions, reuseRowObject: false);

            Console.WriteLine();
            Console.WriteLine("=== Prediction test of loaded model with a multiple samples ===");
            Console.WriteLine();
            IEnumerable<(SentimentData sentiment, SentimentPrediction prediction)> sentimentsAndPredictions = sentiments.Zip(predictedResults, (sentiment, prediction) => (sentiment, prediction));
            foreach ((SentimentData sentiment, SentimentPrediction prediction) item in sentimentsAndPredictions)
            {
                Console.WriteLine($"Sentiment: {item.sentiment.SentimentText} | Prediction: {(Convert.ToBoolean(item.prediction.Prediction) ? "Positive" : "Negative")} | Probability: {item.prediction.Probability} ");

            }
            Console.WriteLine("=== End of predictions ===");
        }

        private static void UseLoadedModelWithUserInput(MLContext mlContext)
        {
            //Load the saved (already trained) model
            ITransformer loadedModel;
            using (var stream = new FileStream(_modelPath, FileMode.Open, FileAccess.Read, FileShare.Read))
            {
                loadedModel = mlContext.Model.Load(stream);
            }

            PredictionEngine<SentimentData, SentimentPrediction> predictionFunction = loadedModel.CreatePredictionEngine<SentimentData, SentimentPrediction>(mlContext);

            Console.WriteLine();
            Console.WriteLine("=== Prediction test of model with user input ===");
            Console.WriteLine();
            Console.WriteLine("Please enter your comment here:");
            string comment = Console.ReadLine();
            SentimentData sentimentStatement = new SentimentData
            {
                SentimentText = comment
            };

            //Predict results for user input
            var resultprediction = predictionFunction.Predict(sentimentStatement);

            Console.WriteLine($"Sentiment: {sentimentStatement.SentimentText} | Prediction: {(Convert.ToBoolean(resultprediction.Prediction) ? "Positive" : "Negative")} | Probability: {resultprediction.Probability} ");

            Console.WriteLine();
            Console.WriteLine();
            Console.WriteLine("Please enter another comment here:");
            string comment2 = Console.ReadLine();
            SentimentData sentimentStatement2 = new SentimentData
            {
                SentimentText = comment2
            };
            //Just another example
            var resultprediction2 = predictionFunction.Predict(sentimentStatement2);

            Console.WriteLine($"Sentiment: {sentimentStatement2.SentimentText} | Prediction: {(Convert.ToBoolean(resultprediction2.Prediction) ? "Positive" : "Negative")} | Probability: {resultprediction2.Probability} ");

            Console.WriteLine("=== End of predictions with user data ===");
            Console.WriteLine();
        }

        private static void SaveModelAsFile(MLContext mlContext, ITransformer model)
        {
            using (var fs = new FileStream(_modelPath, FileMode.Create, FileAccess.Write, FileShare.Write))
                mlContext.Model.Save(model, fs);
            Console.WriteLine("The model is saved to {0}", _modelPath);
        }
    }
}
