import { NextRequest, NextResponse } from 'next/server';

// Use Docker service name when running in Docker, localhost otherwise
const BACKEND_URL = process.env.BACKEND_URL ||
                    process.env.NEXT_PUBLIC_BACKEND_URL ||
                    'http://backend:8000';

export async function POST(request: NextRequest) {
  try {
    const formData = await request.formData();

    console.log('Attempting to connect to:', BACKEND_URL);

    const response = await fetch(`${BACKEND_URL}/analyze-food`, {
      method: 'POST',
      body: formData,
    });

    console.log('Backend response status:', response.status);

    if (!response.ok) {
      const error = await response.json();
      return NextResponse.json(
        { error: error.detail || 'Failed to analyze image' },
        { status: response.status }
      );
    }

    const data = await response.json();
    return NextResponse.json(data);
  } catch (error) {
    console.error('Error forwarding request to backend:', error);
    return NextResponse.json(
      { error: `Failed to connect to backend service at ${BACKEND_URL}. Make sure the backend is running.` },
      { status: 500 }
    );
  }
}