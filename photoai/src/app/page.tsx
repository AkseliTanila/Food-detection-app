"use client";

import type React from "react";

import { useState, useEffect } from "react";
import { Upload, Sparkles, Camera } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";

export default function FoodRecognizer() {
  const [selectedImage, setSelectedImage] = useState<string | null>(null);
  const [isDragging, setIsDragging] = useState(false);
  const [isAnalyzing, setIsAnalyzing] = useState(false);

  const handleFileSelect = (file: File) => {
    if (file && file.type.startsWith("image/")) {
      const reader = new FileReader();
      reader.onload = (e) => {
        setSelectedImage(e.target?.result as string);
      };
      reader.readAsDataURL(file);
    }
  };

  const [message, setMessage] = useState("Loading...");
  useEffect(() => {
    fetch("/api/")
      .then((res) => res.json())
      .then((data) => setMessage(data.message))
      .catch(() => setMessage("Error connecting to backend"));
  }, []);

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

  const handleAnalyze = () => {
    setIsAnalyzing(true);
    // Placeholder for future AI integration
    setTimeout(() => {
      setIsAnalyzing(false);
    }, 2000);
  };

  const handleReset = () => {
    setSelectedImage(null);
    setIsAnalyzing(false);
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
          <h2 className="text-4xl md:text-5xl font-bold mb-4 text-balance text-foreground">
            {message}
          </h2>
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
                <label htmlFor="file-upload">
                  <Button size="lg" className="cursor-pointer">
                    <Upload className="mr-2 h-4 w-4" />
                    Choose File
                  </Button>
                  <input
                    id="file-upload"
                    type="file"
                    accept="image/*"
                    className="hidden"
                    onChange={(e) => {
                      const file = e.target.files?.[0];
                      if (file) handleFileSelect(file);
                    }}
                  />
                </label>
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

              {/* Results Placeholder */}
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
                ) : (
                  <p className="text-muted-foreground">
                    Click "Analyze" to identify the food in your image. AI
                    integration coming soon!
                  </p>
                )}
              </div>

              <div className="flex gap-3">
                <Button
                  onClick={handleAnalyze}
                  disabled={isAnalyzing}
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
