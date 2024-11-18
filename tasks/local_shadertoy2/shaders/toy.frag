#version 430

// Remain of compute shader!
// layout(local_size_x = 32, local_size_y = 32) in;


// layout(binding = 0, rgba8)  uniform image2D resultImage;
// layout(binding = 1, rgba8)  uniform image2D   cupTexture;
layout(binding = 0)            uniform sampler2D cupTexture;
layout(binding = 1)            uniform sampler2D skyboxTexture;
layout(push_constant)       uniform Parameters {
  // GLSL has uints.
  //   https://www.khronos.org/opengl/wiki/Data_Type_(GLSL)#Scalars
  uint iResolution_x;
  uint iResolution_y;
  float time;
} parameters;

float time;
vec2  iResolution;
vec3  mouse = vec3(sin(parameters.time / 2) / 2.0, -0.4, 0);

// Cup of soup in smokes

// ----
// Box mapping by Inigo Quilez from https://www.shadertoy.com/view/MtsGWH

// The MIT License
// Copyright © 2015 Inigo Quilez
// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions: The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software. THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.


// How to do texture map a 3D object when it doesn't have 
// uv coordinates but can't afford full 3D solid texturing.

// The idea is to perform three planar texture projections 
// and blend the results based on the alignment of the
// normal vector to each one of the projection directions.

// The technique was invented by Mitch Prater in the early
// 90s, and has been called "Box mapping" or "Rounded cube
// mapping" traditionally, although more recently it has
// become popular in the realtime rendering community and
// rebranded as "triplanar" mapping.

// For a "biplanar" mapping example, visit:
//
// https://www.shadertoy.com/view/ws3Bzf



// "p" point apply texture to
// "n" normal at "p"
// "k" controls the sharpness of the blending in the
//     transitions areas.
// "s" texture sampler
vec4 boxmap( in sampler2D s, in vec3 p, in vec3 n, in float k )
{
    // В некоторых случаях координаты отрицательные,
    //   тогда текстуру прочитать не получится. По отрицательным координатам там
    //   белый цвет, правильные координаты находятся в отрезке от 0 до 1. Потому
    p = abs(p);

    // project+fetch
	vec4 x = texture( s, p.yz);
	vec4 y = texture( s, p.zx);
	vec4 z = texture( s, p.xy);
    
    // and blend
    vec3 m = pow( abs(n), vec3(k) );
	return (x*m.x + y*m.y + z*m.z) / (m.x + m.y + m.z);
}


//===============================================================================================

const vec3  eye      = vec3 ( 0, 0, 5 );
const vec3  light    = vec3  ( 0.0, 0.0, 10.0 );
const int   maxSteps = 70;
const float eps      = 0.00001;
const float pi    = 3.1415926;

// const vec3  cupClr  = vec3 ( 1, 0.3, 0 );
const vec3 soupClr = vec3(0.6, 0.2, 0);

vec4 skyboxClr(vec3 dir) {
    // return texture(iChannel1, dir);
    // return vec4(0.5, 0.4, 0.3, 0.6);
    // Масштабируем, чтобы бороться с черными полосками в выводе.
    // if (dir.x != 0 && dir.y != 0) {
    dir = normalize(dir);
    if (length(dir.xy) > 0.01) {
        return texture(skyboxTexture, abs(dir.xy) * 0.7);
    // } else if (dir.x != 0 && dir.z != 0) {
     } else if (length(dir.xz) > 0.01) {
        return texture(skyboxTexture, abs(dir.xz) * 0.7);
     } else {
        return texture(skyboxTexture, abs(dir.yz) * 0.7);
    }
}

// Rotation matrix around the X axis.
mat3 rotateX(float theta) {
    float c = cos(theta);
    float s = sin(theta);
    return mat3(
        vec3(1, 0, 0),
        vec3(0, c, -s),
        vec3(0, s, c)
    );
}

// Rotation matrix around the Y axis.
mat3 rotateY(float theta) {
    float c = cos(theta);
    float s = sin(theta);
    return mat3(
        vec3(c, 0, s),
        vec3(0, 1, 0),
        vec3(-s, 0, c)
    );
}

// Rotation matrix around the Z axis.
mat3 rotateZ(float theta) {
    float c = cos(theta);
    float s = sin(theta);
    return mat3(
        vec3(c, s, 0),
        vec3(-s, c, 0),
        vec3(0, 0, 1)
    );
}


float smin ( float a, float b, float k )
{
	float res = exp ( -k*a ) + exp ( -k*b );
	return -log ( res ) / k;
}

float sdCutSphere( vec3 p, float r, float h )
{
  // sampling independent computations (only depend on shape)
  float w = sqrt(r*r-h*h);

  // sampling dependant computations
  vec2 q = vec2( length(p.xz), p.y );
  float s = max( (h-r)*q.x*q.x+w*w*(h+r-2.0*q.y), h*q.x-w*q.y );
  return (s<0.0) ? length(q)-r :
         (q.x<w) ? h - q.y     :
                   length(q-vec2(w,h));
}

// Функция не работает...
float sdDiff(float srcdist, float subdist) {
    // Представим себе летящий луч, пересекающий фигуры.
    // -->->- src ->D>- sub
    //   Норм, возвращаем src.
    // -->->- src = sub
    //   Не понятно :( :( :( Не понятно, беру с лекции.
    // -->->- sub ->D>- dst
    //   Возвращаем dst.
    // Ограничение -- вся фигура sub лежит в src,
    //   т.к. мы продвигаемся по лучу до окончания фигуры
  
   return max(-subdist, srcdist);
}
/*
float sdf ( in vec3 p, in mat3 m )
{
   vec3 q = m * p;
   
   float base = sdDiff(sdCutSphere(q, 1.0, -1.0), sdCutSphere(q, 1.0, 0.3));
   float inner = sdCutSphere(q, 0.9, -0.9);
   float result = sdDiff(base, inner);
   return sdCutSphere(q, 1.0, 0.3);
}
*/


float sdCutHollowSphere( vec3 p, float r, float h, float t )
{
  // sampling independent computations (only depend on shape)
  float w = sqrt(r*r-h*h);
  
  // sampling dependant computations
  vec2 q = vec2( length(p.xz), p.y );
  return ((h*q.x<w*q.y) ? length(q-vec2(w,h)) : 
                          abs(length(q)-r) ) - t;
}

float sdCappedTorus( vec3 p, vec2 sc, float ra, float rb)
{
  p.x = abs(p.x);
  float k = (sc.y*p.x>sc.x*p.y) ? dot(p.xy,sc) : length(p.xy);
  return sqrt( dot(p,p) + ra*ra - 2.0*ra*k ) - rb;
}

float sdPlane( vec3 p, vec3 n, float h )
{
  // n must be normalized
  return dot(p,n) + h;
}

float sdfCup(in vec3 p) {
   //m = mat3(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0);
   vec3 realpnt = p;
   float base = sdCutHollowSphere(realpnt, 1.0, 0.4, 0.1);
   vec3 cuppnt = realpnt;
   
   float result = base;

   // Дополнительно вращаем экран, чтобы получить 
   //   разрвернутую фигуру.
   mat3 rotViewHandle1 = rotateZ(0.5 * pi);
   vec3 offsetViewHandle1 = vec3(0.0, -1.0, 0.0);
   realpnt = rotViewHandle1 * p + offsetViewHandle1;
   float handle1 = sdCappedTorus(realpnt, vec2(1.0, 0.0), 0.3, 0.05);
   result = min(result, handle1);

   // Дополнительно вращаем экран, чтобы получить 
   //   разрвернутую фигуру.
   mat3 rotViewHandle2 = rotateZ(-0.5 * pi);
   vec3 offsetViewHandle2 = vec3(0.0, -1.0, 0.0);
   realpnt = rotViewHandle2 * p + offsetViewHandle2;
   float handle2 = sdCappedTorus(realpnt, vec2(1.0, 0.0), 0.3, 0.05);
   result = min(result, handle2);
   
   return result;
}

vec3 generateNormalForCup ( vec3 z, float d)
{
    // mat3 m = mat3(1, 0, 0, 0, 1, 0, 0, 0, 1);
    float e   = max (d * 0.5, eps );
    vec3 clr = vec3(0.0, 0.0, 0.0);
    float dx1 = sdfCup(z + vec3(e, 0, 0));
    float dx2 = sdfCup(z - vec3(e, 0, 0));
    float dy1 = sdfCup(z + vec3(0, e, 0));
    float dy2 = sdfCup(z - vec3(0, e, 0));
    float dz1 = sdfCup(z + vec3(0, 0, e));
    float dz2 = sdfCup(z - vec3(0, 0, e));
    
    return normalize ( vec3 ( dx1 - dx2, dy1 - dy2, dz1 - dz2 ) );
}


vec4 cupClr(in vec3 p) {
   // return vec4(0.7, 0.4, 0.2, 1.0);
   // return boxmap(iChannel0, realpnt, generateNormalForCup(p, eps, m), 0.3 + 0.7 * (1.0 + sin(pi * time))); 
   // return texture(cupTexture, realpnt.yz);
   return boxmap(cupTexture, p, generateNormalForCup(p, eps), 0.3 + 0.45 * (1.0 + sin(pi * time))); 
}


float sdf ( in vec3 p, in mat3 m, out vec3 clr )
{
   float result = sdfCup(m * p);
   clr = cupClr(m * p).rgb;
   
   mat3 rotViewSoup = rotateX(pi);
   vec3 realpnt = rotViewSoup * m * p;
   // Заполняется снова за 5 секунд.
   float soup = sdCutSphere(realpnt, 1.0, max(0.0, sin(time * 2.0 * pi / 5.0)));
   if (soup < result) {
       clr = soupClr;
   }
   result = min(result, soup);
   
   //realpnt = m * p;
   // (p, n) = d для случая, когда плоскость проходит через ноль,
   //   оттуда же идет радиус-вектор p к точке.
   // (p-p0, n) = d. Теперь плоскость проходит через p0, получаем
   //   радиус-вектор к p после перехода в СО с началом в p0.
   //   А дальше свели задачу к первой.
   // С другой стороны (p-p0, n) = d = (p, n) - (p0, n).
   //   Т.е. h в sdf равна -(p0, n).
   //float skybox_up = abs(sdPlane(realpnt, vec3(0, 1, 0), -6.0));
   //if (skybox_up < result + eps) {
       // Можно считать, что здесь уже точка на плоскости.
       //   Поскольку цвет применяется только когда итерации
       //   raymarching завершились.
       // Плоскость проходит через (0, 6, 0). Координаты
       //   картинки будут (x, z). Другие плоскости обрубают
       //   эту, координаты x и z в видимой части меняются от -6
       //   до 6. Значит, нормированные координаты пикселя в картинке
       //   будут (x + 6) / 12, (z + 6) / 12.
   //    clr = skyboxClr((realpnt.x + 6.0) / 16.0, (realpnt.z + 6.0) / 16.0);
   //}
   // result = min(result, skybox_up);
   /*
   float skybox_down = abs(sdPlane(realpnt, vec3(0, 1, 0), 6.0));
   if (skybox_down < result + eps) {
       clr = skyboxClr((realpnt.x + 6.0) / 16.0, (realpnt.z + 6.0) / 16.0);
   }
   result = min(result, skybox_down);

   float skybox_left = abs(sdPlane(realpnt, vec3(1, 0, 0), -6.0));
   if (skybox_left < result + eps) {
       clr = skyboxClr((realpnt.y + 8.0) / 16.0, (realpnt.z + 6.0) / 16.0);
   }
   result = min(result, skybox_left);

   float skybox_right = abs(sdPlane(realpnt, vec3(1, 0, 0), 3.0));
   if (skybox_right < result + eps) {
       clr = skyboxClr((realpnt.y + 8.0) / 16.0, (realpnt.z + 6.0) / 16.0);
   }
   result = min(result, skybox_left);

   float skybox_front = abs(sdPlane(realpnt, vec3(0, 0, 1), 6.0));
   if (skybox_front < result + eps) {
       clr = skyboxClr((realpnt.x + 8.0) / 16.0, (realpnt.y + 6.0) / 16.0);
   }
   result = min(result, skybox_front);

   float skybox_back = abs(sdPlane(realpnt, vec3(0, 0, 1), -8.0));
   if (skybox_back < result + eps) {
       clr = skyboxClr((realpnt.x + 8.0) / 16.0, (realpnt.y + 8.0) / 16.0);
   }
   result = min(result, skybox_back);
   */


   return result;
}


vec3 trace ( in vec3 from, in vec3 dir, out bool hit, in mat3 m, out vec3 clr )
{
	vec3	p         = from;
	float	totalDist = 0.0;
	
	hit = false;
	
	for ( int steps = 0; steps < maxSteps; steps++ )
	{
		float	dist = sdf ( p, m, clr );
        
		if ( dist < eps )
		{
			hit = true;
			break;
		}
		
		totalDist += dist;
		
		if ( totalDist > 100.0 )
			break;
			
		p += dist * dir;
	}
	
	return p;
}

vec3 generateNormal ( vec3 z, float d, in mat3 m )
{
    float e   = max (d * 0.5, eps );
    vec3 clr = vec3(0.0, 0.0, 0.0);
    float dx1 = sdf(z + vec3(e, 0, 0), m, clr);
    float dx2 = sdf(z - vec3(e, 0, 0), m, clr);
    float dy1 = sdf(z + vec3(0, e, 0), m, clr);
    float dy2 = sdf(z - vec3(0, e, 0), m, clr);
    float dz1 = sdf(z + vec3(0, 0, e), m, clr);
    float dz2 = sdf(z - vec3(0, 0, e), m, clr);
    
    return normalize ( vec3 ( dx1 - dx2, dy1 - dy2, dz1 - dz2 ) );
}

const float roughness = 0.2;
const vec3  r0   = vec3 ( 1.0, 0.92, 0.23 );
const float gamma = 2.2;
const float FDiel = 0.04;		// Fresnel for dielectrics

vec3 fresnel ( in vec3 f0, in float product )
{
	product = clamp ( product, 0.0, 1.0 );		// saturate
	
	return mix ( f0, vec3 (1.0), pow(1.0 - product, 5.0) );
}

float D_blinn(in float roughness, in float NdH)
{
    float m = roughness * roughness;
    float m2 = m * m;
    float n = 2.0 / m2 - 2.0;
    return (n + 2.0) / (2.0 * pi) * pow(NdH, n);
}

float D_beckmann ( in float roughness, in float NdH )
{
	float m    = roughness * roughness;
	float m2   = m * m;
	float NdH2 = NdH * NdH;
	
	return exp( (NdH2 - 1.0) / (m2 * NdH2) ) / (pi * m2 * NdH2 * NdH2);
}

float D_GGX ( in float roughness, in float NdH )
{
	float m  = roughness * roughness;
	float m2 = m * m;
	float NdH2 = NdH * NdH;
	float d  = (m2 - 1.0) * NdH2 + 1.0;
	
	return m2 / (pi * d * d);
}

float G_schlick ( in float roughness, in float nv, in float nl )
{
    float k = roughness * roughness * 0.5;
    float V = nv * (1.0 - k) + k;
    float L = nl * (1.0 - k) + k;
	
    return 0.25 / (V * L);
}

float G_neumann ( in float nl, in float nv )
{
	return nl * nv / max ( nl, nv );
}

float G_klemen ( in float nl, in float nv, in float vh )
{
	return nl * nv / (vh * vh );
}

float G_default ( in float nl, in float nh, in float nv, in float vh )
{
	return min ( 1.0, min ( 2.0*nh*nv/vh, 2.0*nh*nl/vh ) );
}

vec4 cookTorrance ( in vec3 p, in vec3 n, in vec3 l, in vec3 v, in vec3 clr )
{
    vec3  h    = normalize ( l + v );
	float nh   = dot (n, h);
	float nv   = dot (n, v);
	float nl   = dot (n, l);
	float vh   = dot (v, h);
    float metallness = 1.0;
    vec3  base  = pow ( clr, vec3 ( gamma ) );
    vec3  F0    = mix ( vec3(FDiel), clr, metallness );
	
			// compute Beckman
   	float d = D_beckmann ( roughness, nh );

            // compute Fresnel
    vec3 f = fresnel ( F0, nv );
	
            // default G
    float g = G_default ( nl, nh, nv, vh );
	
			// resulting color
	vec3  ct   = f*(0.25 * d * g / nv);
	vec3  diff = max(nl, 0.0) * ( vec3 ( 1.0 ) - f ) / pi;
	float ks   = 0.5;

	return vec4 ( pow ( diff * base + ks * ct, vec3 ( 1.0 / gamma ) ), 1.0 );
}

void mainImage( out vec4 fragColor, in ivec2 iFragCoord )
{

    // Normalized pixel coordinates (from 0 to 1)
    //vec3 mouse = vec3(iMouse.xy/iResolution.xy - 0.5,iMouse.z-.5);

    // Повернем кружку дном вниз. Просто добавим к mouse.x так,
    //   чтобы картинка повернулась по y на pi.
    //   Поведение прокрутки не изменится, почти все значения
    //   синуса все так же будут пробегаться (все бы, почти все,
    //   пробегались, если бы была формула pi * mouse.y).
    // mouse.y += pi / 6.0; // Повернем кружку дном вниз :)

    mat3 m     = rotateX ( 6.0*mouse.y ) * rotateY ( 6.0*mouse.x);
    
    
    vec2 scale = 8.0 * iResolution.xy / max ( iResolution.x, iResolution.y ) ;
    vec2 uv    = scale * ( iFragCoord/iResolution.xy - vec2 ( 0.5 ) );
	vec3 dir   = normalize ( vec3 ( uv, 0 ) - eye );
    vec4 color = vec4 ( 0, 0, 0, 1 );
    
    vec3 clr = vec3(0, 0, 0);
    bool hit;
	vec3 p     = trace ( eye, dir, hit, m, clr );


	if ( hit )
	{
		vec3  l  = normalize        ( light - p );
        vec3  v  = normalize        ( eye - p );
		vec3  n  = generateNormal   ( p, 0.001, m );
		float nl = max ( 0.0, dot ( n, l ) );
        vec3  h  = normalize ( l + v );
        float hn = max ( 0.0, dot ( h, n ) );
        float sp = pow ( hn, 150.0 );
		
        color = cookTorrance ( p, n, l, v, clr );
       
        // Output to screen
        fragColor = color * 3.0;
	} else {
        // https://inspirnathan.com/posts/63-shadertoy-tutorial-part-16
        fragColor = skyboxClr(m * dir);
    }
    
    // fragColor = texture(skyboxTexture, iFragCoord / iResolution);
    // Отладочный вывод показал, что просто есть черные полоски в
    //   процедурной текстуре. Почему не отмасштабировалось -- не ясно,
    //   уже не буду разбираться..
}

// Needed for fragment shader. Now we compute color per pixel!
//   That's just what we need from the glsl. No more whole
//   new image to put data to.
// Also fragment shader seems to think in float coordinates.
//   Not in integer ones. Because it's better for scaling,
//   maybe. And previously we worked with a raw image.
// layout (location = 0) in  vec2 fragCoord;
layout (location = 0) out vec4 fragColor;

void main()
{
  iResolution = vec2(parameters.iResolution_x, parameters.iResolution_y);
  time = parameters.time;
  // time = 0;

  // Remains of compute shader!
  // ---------------------------------------------------------------
  // vec4 fragColor;
  // vec2  fragCoord  = vec2(gl_GlobalInvocationID.xy);
  // ivec2 iFragCoord = ivec2(gl_GlobalInvocationID.xy);
  // mainImage(fragColor, iFragCoord);

  // Screen initial size! Resizing is not supported.
  // ivec2 screenSize = imageSize(resultImage);
  // if (iFragCoord.x < screenSize.x && iFragCoord.y < screenSize.y)
  //   imageStore(resultImage, iFragCoord, fragColor);
  // ---------------------------------------------------------------

  // In fragment shader, we just write the color onto the image.
  // Вот тут есть встроенные переменные GLSL. Если нас просят
  //   цвет для float пикселя (семплирование без нас делают,
  //   пиксели транслируют во флоты), то и координату должны давать!
  //   И у нас еще есть вершинный шейдер, все может быть связано
  //   с gl_Position, который он проставляет, еще и поэтому решил загуглить.
  //   https://learnopengl.com/Advanced-OpenGL/Advanced-GLSL
  //   https://stackoverflow.com/questions/18590074/glsl-passing-coordinates-to-fragment-shader
  // vec2 fragCoord = gl_FragCoord.xy; // Это тоже не работает, тут, видимо, координаты не от 0 до 1.
  // Проверяем.
  // if (fragCoord.x < 100 && fragCoord.y < 100) {
  //  fragColor = vec4(0, 0, 0, 0);
  // } else
  // Видим маленький квадратик. А если бы координаты были от 0 до 1, то весь экран был бы черным.
  //   Вот в любом непонятном случае главное понять, как надо будет отлаживать. Поэтапно, упростив
  //   алгоритм.
  // https://www.khronos.org/opengl/wiki/Fragment_Shader#Inputs
  // vec2 fragCoord = gl_PointCoord;
  // if (fragCoord.x < 100 && fragCoord.y < 100) {
  //   fragColor = vec4(0, 0, 0, 0);
  // } else
  // Вот это уже лучше работает, да, т.к. уже весь экран черный, а не маленький квадрат.
  //   Ну и правда, это координаты точки, вещественные, а не координаты внутри
  //   фрагмента, т.е. как бы картинки.
  //  if (fragCoord.x < 0.1 && fragCoord.y < 0.1) {
  //   fragColor = vec4(0, 0, 0, 0);
  //  } else
  // проблема в том, что эта переменная, видимо, стабильно равна нулю..
  // fragCoord = gl_PointCoord; // Не работает.
  // fragCoord = vec2(gl_FragCoord.x / iResolution.x, gl_FragCoord.y / iResolution.y);
  //  if (fragCoord.x < 1 && fragCoord.y < 0.1) {
  //   fragColor = vec4(0, 0, 0, 0);
  //  } else
  // Работает.
  // Другой вариант.
  ivec2 iFragCoord = ivec2(gl_FragCoord.xy);
  mainImage(fragColor, iFragCoord);

  // if (sin(time) < 0) {
  //   fragColor = vec4(0, 0, 0, 0);
  // }

  // Debugging coordinate, if we get them at all!
  //if (fragCoord.x < 0.1 && fragCoord.y < 0.1 && time < 5) {
  //  fragColor = vec4(0, 0, 0, 0);
  //}

  // Debugging to check that cupTexture is loaded.
  //   cupTexture should be said to have type image2D before this!
  // ivec2 textureSize = imageSize(cupTexture);
  // if (iFragCoord.x < textureSize.x && iFragCoord.y < textureSize.y) {
  //   imageStore(resultImage, iFragCoord, imageLoad(cupTexture, iFragCoord));
  // }

  // Debugging to check if we get sampler2D.
  //   cupTexture should be said to have type sampler2D before this!
  // For compute shader.
  // imageStore(resultImage, iFragCoord, texture(cupTexture, iFragCoord / iResolution.xy));
  // For fragment shader.
  // fragColor = texture(cupTexture, iFragCoord / iResolution.xy);
}
