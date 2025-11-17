"use client";

import type React from "react";

import { useState, useEffect, useRef } from "react";
import { Upload, Sparkles, Camera } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";

interface FoodItem {
  name: string;
  confidence: number;
  calories: number;
  protein_g: number;
  carbs_g: number;
  fat_g: number;
}

interface AnalysisResult {
  success: boolean;
  filename: string;
  size_bytes: number;
  analysis: {
    detected_foods: FoodItem[];
    total_nutrition: {
      calories: number;
      protein_g: number;
      carbs_g: number;
      fat_g: number;
    };
    processing_time_ms: number;
    model_version: string;
  };
}

export default function FoodRecognizer() {
  const [selectedImage, setSelectedImage] = useState<string | null>(null);
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [isDragging, setIsDragging] = useState(false);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [analysisResult, setAnalysisResult] = useState<AnalysisResult | null>(
    null
  );
  const [error, setError] = useState<string | null>(null);
  const fileInputRef = useRef<HTMLInputElement | null>(null);

  const handleFileSelect = (file: File) => {
    if (file && file.type.startsWith("image/")) {
      setSelectedFile(file);
      const reader = new FileReader();
      reader.onload = (e) => {
        setSelectedImage(e.target?.result as string);
      };
      reader.readAsDataURL(file);
      setAnalysisResult(null);
      setError(null);
    }
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
    const file = e.dataTransfer.files[0];
    handleFileSelect(file);
  };

  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(true);
  };

  const handleDragLeave = () => {
    setIsDragging(false);
  };

  const handleAnalyze = async () => {
    if (!selectedFile) return;

    setIsAnalyzing(true);
    setError(null);

    try {
      const formData = new FormData();
      formData.append("file", selectedFile);

      const response = await fetch("/api/analyze-food", {
        method: "POST",
        body: formData,
      });

      const data = await response.json();

      if (!response.ok) {
        throw new Error(data.error || "Failed to analyze image");
      }

      setAnalysisResult(data);
    } catch (err) {
      setError(err instanceof Error ? err.message : "An error occurred");
      console.error("Analysis error:", err);
    } finally {
      setIsAnalyzing(false);
    }
  };

  const handleReset = () => {
    setSelectedImage(null);
    setSelectedFile(null);
    setIsAnalyzing(false);
    setAnalysisResult(null);
    setError(null);
  };

  return (
    <div className="min-h-screen bg-background">
      {/* Header */}
      <header className="border-b border-border">
        <div className="container mx-auto px-4 py-4 flex items-center justify-between">
          <div className="flex items-center gap-2">
            <Camera className="h-6 w-6 text-primary" />
            <h1 className="text-xl font-semibold text-foreground">Foodys</h1>
          </div>
          <p className="text-sm text-muted-foreground hidden sm:block">
            AI food detection
          </p>
        </div>
      </header>

      {/* Main Content */}
      <main className="container mx-auto px-4 py-12 max-w-4xl">
        <div className="text-center mb-12">
          <h2 className="text-4xl md:text-5xl font-bold mb-4 text-balance text-foreground"></h2>
          <p className="text-lg text-muted-foreground max-w-2xl mx-auto text-pretty">
            Upload a photo and let our AI recognize what food it is. Get instant
            results with detailed information.
          </p>
        </div>

        <Card className="p-8 bg-card border-border">
          {!selectedImage ? (
            <div
              onDrop={handleDrop}
              onDragOver={handleDragOver}
              onDragLeave={handleDragLeave}
              className={`border-2 border-dashed rounded-lg p-12 text-center transition-colors ${
                isDragging
                  ? "border-primary bg-primary/5"
                  : "border-border hover:border-primary/50"
              }`}
            >
              <div className="flex flex-col items-center gap-4">
                <div className="rounded-full bg-primary/10 p-6">
                  <Upload className="h-12 w-12 text-primary" />
                </div>
                <div>
                  <h3 className="text-xl font-semibold mb-2 text-card-foreground">
                    Upload a food image
                  </h3>
                  <p className="text-muted-foreground mb-4">
                    Drag and drop your image here, or click to browse
                  </p>
                </div>
                <Button
                  size="lg"
                  type="button"
                  className="cursor-pointer"
                  onClick={() => fileInputRef.current?.click()}
                >
                  <Upload className="mr-2 h-4 w-4" />
                  Choose File
                </Button>
                <input
                  id="file-upload"
                  ref={fileInputRef}
                  type="file"
                  accept="image/*"
                  className="sr-only"
                  onChange={(e) => {
                    const file = e.target.files?.[0];
                    if (file) handleFileSelect(file);
                  }}
                />
                <p className="text-xs text-muted-foreground">
                  Supports JPG, PNG, WEBP up to 10MB
                </p>
              </div>
            </div>
          ) : (
            <div className="space-y-6">
              <div className="relative rounded-lg overflow-hidden bg-muted">
                <img
                  src={selectedImage || "/placeholder.svg"}
                  alt="Selected food"
                  className="w-full h-auto max-h-96 object-contain"
                />
              </div>

              {/* Error Message */}
              {error && (
                <div className="bg-red-50 dark:bg-red-950 border border-red-200 dark:border-red-800 rounded-lg p-4">
                  <p className="text-red-800 dark:text-red-200 text-sm">
                    {error}
                  </p>
                </div>
              )}

              {/* Results */}
              <div className="bg-secondary/50 rounded-lg p-6 border border-border">
                <div className="flex items-center gap-2 mb-4">
                  <Sparkles className="h-5 w-5 text-accent" />
                  <h3 className="font-semibold text-card-foreground">
                    AI Analysis
                  </h3>
                </div>
                {isAnalyzing ? (
                  <div className="flex items-center gap-3">
                    <div className="animate-spin rounded-full h-5 w-5 border-2 border-primary border-t-transparent" />
                    <p className="text-muted-foreground">
                      Analyzing your image...
                    </p>
                  </div>
                ) : analysisResult ? (
                  <div className="space-y-4">
                    <div className="space-y-3">
                      {analysisResult.analysis.detected_foods.map(
                        (food, index) => (
                          <div
                            key={index}
                            className="bg-background rounded-lg p-4"
                          >
                            <div className="flex justify-between items-start mb-2">
                              <h4 className="font-medium text-foreground">
                                {food.name}
                              </h4>
                              <span className="text-sm text-muted-foreground">
                                {(food.confidence * 100).toFixed(0)}% confidence
                              </span>
                            </div>

                            {/* Only show nutrition if available */}
                            {food.calories ? (
                              <div className="grid grid-cols-4 gap-2 text-sm">
                                <div>
                                  <p className="text-muted-foreground">
                                    Calories
                                  </p>
                                  <p className="font-medium">{food.calories}</p>
                                </div>
                                <div>
                                  <p className="text-muted-foreground">
                                    Protein
                                  </p>
                                  <p className="font-medium">
                                    {food.protein_g}g
                                  </p>
                                </div>
                                <div>
                                  <p className="text-muted-foreground">Carbs</p>
                                  <p className="font-medium">{food.carbs_g}g</p>
                                </div>
                                <div>
                                  <p className="text-muted-foreground">Fat</p>
                                  <p className="font-medium">{food.fat_g}g</p>
                                </div>
                              </div>
                            ) : (
                              <p className="text-sm text-muted-foreground">
                                Nutrition data not available
                              </p>
                            )}
                          </div>
                        )
                      )}
                    </div>
                    <div className="bg-primary/10 rounded-lg p-4">
                      <h4 className="font-semibold mb-2 text-foreground">
                        Total Nutrition
                      </h4>
                      <div className="grid grid-cols-4 gap-2 text-sm">
                        {analysisResult.analysis.total_nutrition ? (
                          <>
                            <div>
                              <p className="text-muted-foreground">Calories</p>
                              <p className="font-bold text-primary">
                                {
                                  analysisResult.analysis.total_nutrition
                                    .calories
                                }
                              </p>
                            </div>
                            <div>
                              <p className="text-muted-foreground">Protein</p>
                              <p className="font-bold text-primary">
                                {
                                  analysisResult.analysis.total_nutrition
                                    .protein_g
                                }
                                g
                              </p>
                            </div>
                            <div>
                              <p className="text-muted-foreground">Carbs</p>
                              <p className="font-bold text-primary">
                                {
                                  analysisResult.analysis.total_nutrition
                                    .carbs_g
                                }
                                g
                              </p>
                            </div>
                            <div>
                              <p className="text-muted-foreground">Fat</p>
                              <p className="font-bold text-primary">
                                {analysisResult.analysis.total_nutrition.fat_g}g
                              </p>
                            </div>
                          </>
                        ) : (
                          <p className="text-sm text-muted-foreground">
                            Nutrition data not available
                          </p>
                        )}
                      </div>
                    </div>
                  </div>
                ) : (
                  <p className="text-muted-foreground">
                    Click "Analyze" to identify the food in your image
                  </p>
                )}
              </div>

              <div className="flex gap-3">
                <Button
                  onClick={handleAnalyze}
                  disabled={isAnalyzing || !selectedFile}
                  size="lg"
                  className="flex-1"
                >
                  {isAnalyzing ? (
                    <>
                      <div className="animate-spin rounded-full h-4 w-4 border-2 border-primary-foreground border-t-transparent mr-2" />
                      Analyzing...
                    </>
                  ) : (
                    <>
                      <Sparkles className="mr-2 h-4 w-4" />
                      Analyze Food
                    </>
                  )}
                </Button>
                <Button
                  onClick={handleReset}
                  variant="outline"
                  size="lg"
                  disabled={isAnalyzing}
                >
                  Upload New
                </Button>
              </div>
            </div>
          )}
        </Card>

        {/* Features */}
        <div className="grid md:grid-cols-3 gap-6 mt-12">
          <div className="text-center p-6">
            <div className="rounded-full bg-primary/10 w-12 h-12 flex items-center justify-center mx-auto mb-4">
              <Sparkles className="h-6 w-6 text-primary" />
            </div>
            <h3 className="font-semibold mb-2 text-foreground">AI-Powered</h3>
            <p className="text-sm text-muted-foreground">
              Advanced machine learning for accurate food identification
            </p>
          </div>
          <div className="text-center p-6">
            <div className="rounded-full bg-accent/10 w-12 h-12 flex items-center justify-center mx-auto mb-4">
              <Camera className="h-6 w-6 text-accent" />
            </div>
            <h3 className="font-semibold mb-2 text-foreground">
              Instant Results
            </h3>
            <p className="text-sm text-muted-foreground">
              Get food recognition results in seconds
            </p>
          </div>
          <div className="text-center p-6">
            <div className="rounded-full bg-primary/10 w-12 h-12 flex items-center justify-center mx-auto mb-4">
              <Upload className="h-6 w-6 text-primary" />
            </div>
            <h3 className="font-semibold mb-2 text-foreground">Easy Upload</h3>
            <p className="text-sm text-muted-foreground">
              Drag and drop or click to upload your food photos
            </p>
          </div>
        </div>
      </main>
    </div>
  );
}
