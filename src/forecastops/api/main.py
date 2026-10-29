"""FastAPI application for ForecastOps API."""

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from forecastops.api.routes import router
from forecastops.utils.logger import setup_logger

logger = setup_logger(__name__)

@asynccontextmanager
async def lifespan(app):
    logger.info("ForecastOps API starting up")
    yield
    logger.info("ForecastOps API shutting down")


# Create FastAPI app
app = FastAPI(
    lifespan=lifespan,
    title="ForecastOps API",
    description="Retail Demand Forecasting & Optimization Platform",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", include_in_schema=False)
def root():
    """Root endpoint."""
    return {
        "name": "ForecastOps API",
        "version": "1.0.0",
        "docs": "/docs",
        "status": "running"
    }


@app.get("/health", include_in_schema=False)
def health():
    """Health check endpoint."""
    return {"status": "healthy"}


@app.exception_handler(Exception)
async def global_exception_handler(request, exc: Exception):
    """Global exception handler."""
    logger.error(f"Global exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error"}
    )


# Include routes
app.include_router(router)
