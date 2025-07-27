#!/bin/bash

# Script para construir y subir imagen Docker a AWS ECR
# Para ML Santiago API - Versión optimizada para tamaño

set -e

# Configuración (modificar según tu configuración)
AWS_REGION="${AWS_REGION:-us-east-2}"
ECR_REPOSITORY="${ECR_REPOSITORY:-ml-santiago-api}"
IMAGE_TAG="${IMAGE_TAG:-latest}"
DOCKERFILE="${DOCKERFILE:-Dockerfile.lightweight}"  # Usar versión optimizada por defecto

# Colores para output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

print_info() {
    echo -e "${BLUE}ℹ️  $1${NC}"
}

print_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

# Función para mostrar ayuda
show_help() {
    echo "🐳 Script para construir y subir ML Santiago API a ECR"
    echo ""
    echo "Uso: $0 [opciones]"
    echo ""
    echo "Opciones:"
    echo "  -r, --region REGION        Región de AWS (default: us-east-1)"
    echo "  -n, --repository NAME      Nombre del repositorio ECR (default: ml-santiago-api)"
    echo "  -t, --tag TAG             Tag de la imagen (default: latest)"
    echo "  -f, --dockerfile FILE      Dockerfile a usar (default: Dockerfile.lightweight)"
    echo "  --build-only              Solo construir, no subir a ECR"
    echo "  --dev                     Construir imagen de desarrollo"
    echo "  --standard                Usar Dockerfile estándar (no optimizado)"
    echo "  --distroless             Usar imagen distroless (ultra-ligera)"
    echo "  -h, --help                Mostrar esta ayuda"
    echo ""
    echo "Variables de entorno:"
    echo "  AWS_REGION                Región de AWS"
    echo "  ECR_REPOSITORY            Nombre del repositorio ECR"
    echo "  IMAGE_TAG                 Tag de la imagen"
    echo ""
    echo "Ejemplos:"
    echo "  $0                        # Construir y subir con configuración por defecto"
    echo "  $0 --tag v1.0.0          # Usar tag específico"
    echo "  $0 --build-only          # Solo construir localmente"
    echo "  $0 --dev                 # Construir imagen de desarrollo"
    echo "  $0 --standard            # Usar Dockerfile estándar"
    echo "  $0 --distroless          # Usar imagen ultra-ligera"
}

# Función para verificar requisitos
check_requirements() {
    print_info "Verificando requisitos..."
    
    # Verificar Docker
    if ! command -v docker &> /dev/null; then
        print_error "Docker no está instalado"
        exit 1
    fi
    
    # Verificar Docker Buildx
    if ! docker buildx version &> /dev/null; then
        print_warning "Docker Buildx no está disponible, intentando habilitar..."
        docker buildx create --use --name multiarch-builder || true
    fi
    
    # Verificar AWS CLI
    if ! command -v aws &> /dev/null; then
        print_error "AWS CLI no está instalado"
        exit 1
    fi
    
    # Verificar que Docker esté corriendo
    if ! docker info &> /dev/null; then
        print_error "Docker no está ejecutándose"
        exit 1
    fi
    
    print_success "Todos los requisitos están disponibles"
}

# Función para construir imagen
build_image() {
    local target="${TARGET_STAGE:-${1:-production}}"
    
    print_info "Construyendo imagen Docker para linux/amd64..."
    print_info "Dockerfile: $DOCKERFILE"
    print_info "Target: $target"
    print_info "Tag: $IMAGE_TAG"
    
    # Verificar que el modelo existe
    if [ ! -f "models/iris_classifier.h5" ]; then
        print_warning "Modelo no encontrado, entrenando..."
        python train_iris_model.py
    fi
    
    # Crear tag completo
    local full_tag="$ECR_REPOSITORY:$IMAGE_TAG"
    
    # Construir imagen forzando arquitectura AMD64 para ECR
    docker buildx build \
        --platform linux/amd64 \
        --target "$target" \
        --tag "$full_tag" \
        --file "$DOCKERFILE" \
        --build-arg BUILDKIT_INLINE_CACHE=1 \
        --load \
        .
    
    if [ $? -eq 0 ]; then
        print_success "Imagen construida exitosamente: $full_tag"
        
        # Mostrar tamaño de la imagen
        local image_size=$(docker images "$full_tag" --format "table {{.Size}}" | tail -n 1)
        print_info "Tamaño de la imagen: $image_size"
        
        # Mostrar arquitectura de la imagen
        local image_arch=$(docker inspect "$full_tag" --format='{{.Architecture}}')
        print_info "Arquitectura de la imagen: $image_arch"
        
        # Mostrar optimizaciones aplicadas
        if [[ "$DOCKERFILE" == *"lightweight"* ]]; then
            print_success "✨ Optimizaciones aplicadas: imagen ligera para ECR"
        fi
    else
        print_error "Error construyendo la imagen"
        exit 1
    fi
}

# Función para configurar ECR
setup_ecr() {
    print_info "Configurando ECR..."
    
    # Obtener URI completo del repositorio
    ECR_URI="${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com/${ECR_REPOSITORY}"
    
    # Verificar si el repositorio existe
    if ! aws ecr describe-repositories --repository-names "$ECR_REPOSITORY" --region "$AWS_REGION" &> /dev/null; then
        print_warning "Repositorio $ECR_REPOSITORY no existe en ECR"
        read -p "¿Quieres crearlo? (y/N): " create_repo
        
        if [[ $create_repo =~ ^[Yy]$ ]]; then
            print_info "Creando repositorio ECR..."
            aws ecr create-repository \
                --repository-name "$ECR_REPOSITORY" \
                --region "$AWS_REGION" \
                --image-scanning-configuration scanOnPush=true
            print_success "Repositorio creado: $ECR_REPOSITORY"
        else
            print_error "Repositorio requerido para continuar"
            exit 1
        fi
    fi
    
    print_success "Repositorio ECR configurado: $ECR_REPOSITORY"
}

# Función para autenticar con ECR
ecr_login() {
    print_info "Autenticando con ECR..."
    
    # Obtener ID de cuenta AWS
    AWS_ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
    
    if [ -z "$AWS_ACCOUNT_ID" ]; then
        print_error "No se pudo obtener el ID de cuenta AWS. Verifica tu configuración."
        exit 1
    fi
    
    # Login a ECR
    aws ecr get-login-password --region "$AWS_REGION" | \
        docker login --username AWS --password-stdin "${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com"
    
    if [ $? -eq 0 ]; then
        print_success "Autenticación exitosa con ECR"
    else
        print_error "Error en autenticación con ECR"
        exit 1
    fi
}

# Función para tagear y subir imagen
push_image() {
    print_info "Subiendo imagen a ECR..."
    
    # Crear tags
    local local_tag="$ECR_REPOSITORY:$IMAGE_TAG"
    local ecr_tag="${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com/${ECR_REPOSITORY}:${IMAGE_TAG}"
    
    # Tagear imagen para ECR
    docker tag "$local_tag" "$ecr_tag"
    
    # Subir imagen
    docker push "$ecr_tag"
    
    if [ $? -eq 0 ]; then
        print_success "Imagen subida exitosamente a ECR"
        print_success "URI de la imagen: $ecr_tag"
        
        # También tagear como latest si no es latest
        if [ "$IMAGE_TAG" != "latest" ]; then
            local latest_tag="${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com/${ECR_REPOSITORY}:latest"
            docker tag "$local_tag" "$latest_tag"
            docker push "$latest_tag"
            print_success "También taggeado como: $latest_tag"
        fi
    else
        print_error "Error subiendo imagen a ECR"
        exit 1
    fi
}

# Función principal
main() {
    print_info "🚀 Iniciando proceso de build y deploy para ML Santiago API"
    print_info "Región: $AWS_REGION"
    print_info "Repositorio: $ECR_REPOSITORY"
    print_info "Tag: $IMAGE_TAG"
    echo ""
    
    check_requirements
    
    # Construir imagen
    if [ "$DEV_MODE" = true ]; then
        build_image "development"
    else
        build_image "production"
    fi
    
    # Si solo build, terminar aquí
    if [ "$BUILD_ONLY" = true ]; then
        print_success "✨ Build completado (solo local)"
        exit 0
    fi
    
    # Proceso ECR
    ecr_login
    setup_ecr
    push_image
    
    print_success "🎉 ¡Proceso completado exitosamente!"
    print_info "Tu imagen está disponible en ECR y lista para deploy"
}

# Procesar argumentos
BUILD_ONLY=false
DEV_MODE=false

while [[ $# -gt 0 ]]; do
    case $1 in
        -r|--region)
            AWS_REGION="$2"
            shift 2
            ;;
        -n|--repository)
            ECR_REPOSITORY="$2"
            shift 2
            ;;
        -t|--tag)
            IMAGE_TAG="$2"
            shift 2
            ;;
        -f|--dockerfile)
            DOCKERFILE="$2"
            shift 2
            ;;
        --build-only)
            BUILD_ONLY=true
            shift
            ;;
        --dev)
            DEV_MODE=true
            shift
            ;;
        --standard)
            DOCKERFILE="Dockerfile"
            shift
            ;;
        --distroless)
            DOCKERFILE="Dockerfile.lightweight"
            TARGET_STAGE="runtime"
            shift
            ;;
        -h|--help)
            show_help
            exit 0
            ;;
        *)
            print_error "Opción desconocida: $1"
            show_help
            exit 1
            ;;
    esac
done

# Ejecutar función principal
main
