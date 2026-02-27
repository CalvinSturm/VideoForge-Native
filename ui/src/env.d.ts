/// <reference types="vite/client" />

/**
 * CSS module declarations — allows `import "*.css"` without TS2307.
 *
 * Vite handles CSS bundling at build time; these declarations just
 * tell the TypeScript compiler the imports are valid.
 */
declare module "*.css" {
    const content: string;
    export default content;
}
