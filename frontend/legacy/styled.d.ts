import 'styled-components';
import type { BloombergThemeType } from './themes/bloomberg';

declare module 'styled-components' {
  export interface DefaultTheme extends BloombergThemeType {}
}
